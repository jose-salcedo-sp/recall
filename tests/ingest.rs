//! `recall::ingest::ingest` against Postgres: versioning, superseding, and the embedding-model check.

#![allow(
    clippy::unwrap_used,
    reason = "integration-test helpers; clippy exempts only #[test] functions"
)]

use std::sync::Mutex;

use async_trait::async_trait;
use chrono::{DateTime, TimeZone, Utc};
use recall::{
    domain::{ChunkDraft, Document, Labels, Sensitivity, SourceName},
    ingest::{
        EmbeddingModelMismatch, Source, Sources,
        fakes::{FailingSource, StaticSource},
        ingest, markdown,
    },
    llm::{EmbedPurpose, Embedder, ModelRef, fakes::HashEmbedder},
    store::chunks::{ChunkStore, Stored},
};
use sqlx::PgPool;

/// `HashEmbedder` vectors under a configurable model name, recording every batch it embeds.
struct RecordingEmbedder {
    model: ModelRef,
    batches: Mutex<Vec<Vec<String>>>,
}

impl RecordingEmbedder {
    fn new() -> Self {
        Self::named(HashEmbedder.model())
    }

    fn named(model: ModelRef) -> Self {
        Self {
            model,
            batches: Mutex::new(Vec::new()),
        }
    }

    fn batches(&self) -> Vec<Vec<String>> {
        self.batches.lock().unwrap().clone()
    }
}

#[async_trait]
impl Embedder for RecordingEmbedder {
    fn model(&self) -> ModelRef {
        self.model.clone()
    }

    async fn embed(&self, texts: &[String], purpose: EmbedPurpose) -> eros::Result<Vec<Vec<f32>>> {
        assert_eq!(purpose, EmbedPurpose::Document);
        self.batches.lock().unwrap().push(texts.to_vec());
        HashEmbedder.embed(texts, purpose).await
    }
}

fn updated(day: u32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 1, day, 0, 0, 0).unwrap()
}

fn document(source: &SourceName, external_id: &str, chunks: &[(&str, &str)]) -> Document {
    Document {
        source: source.clone(),
        external_id: external_id.into(),
        title: format!("Doc {external_id}"),
        url: format!("https://example.test/{external_id}"),
        updated_at: updated(1),
        labels: Labels {
            acl: vec!["group:arm".into()],
            tenant: None,
            sensitivity: Sensitivity::Internal,
        },
        chunks: chunks
            .iter()
            .map(|(locator, text)| ChunkDraft {
                locator: (*locator).into(),
                heading: Some(locator.to_uppercase()),
                url: format!("https://example.test/{external_id}#{locator}"),
                text: (*text).into(),
                authoritative: true,
            })
            .collect(),
    }
}

fn page(chunks: &[(&str, &str)]) -> Document {
    document(&markdown::NAME, "roles.md", chunks)
}

fn wiki() -> SourceName {
    SourceName::new("wiki").unwrap()
}

fn static_source(documents: Vec<Document>) -> Box<dyn Source> {
    Box::new(StaticSource {
        name: documents[0].source.clone(),
        documents,
    })
}

fn sources(sources: Vec<Box<dyn Source>>) -> Sources {
    Sources::new(sources).unwrap()
}

async fn count(pool: &PgPool, filter: &str) -> i64 {
    sqlx::query_scalar(sqlx::AssertSqlSafe(format!(
        "SELECT count(*) FROM chunks WHERE {filter}"
    )))
    .fetch_one(pool)
    .await
    .unwrap()
}

/// `(id, ordinal, embedding_model)` of current chunks, ordered by id.
async fn current(pool: &PgPool) -> Vec<(String, i32, String)> {
    sqlx::query_as(
        "SELECT id, ordinal, embedding_model FROM chunks WHERE superseded_at IS NULL ORDER BY id",
    )
    .fetch_all(pool)
    .await
    .unwrap()
}

fn stored(new_versions: u64, superseded: u64) -> Stored {
    Stored {
        new_versions,
        superseded,
    }
}

#[sqlx::test(migrator = "recall::store::MIGRATOR")]
async fn first_ingest_inserts_every_chunk(pool: PgPool) {
    let store = ChunkStore::new(pool.clone());
    let embedder = RecordingEmbedder::new();

    let result = store
        .store(&page(&[("a", "alpha"), ("b", "beta")]), &embedder)
        .await
        .unwrap();

    assert_eq!(result, stored(2, 0));
    assert_eq!(
        current(&pool).await,
        [
            ("markdown:roles.md#a".into(), 0, "fake-hash-embedder".into()),
            ("markdown:roles.md#b".into(), 1, "fake-hash-embedder".into()),
        ]
    );
    assert_eq!(
        embedder.batches(),
        [["Doc roles.md\nA\n\nalpha", "Doc roles.md\nB\n\nbeta"]]
    );
    let title: String =
        sqlx::query_scalar("SELECT title FROM documents WHERE id = 'markdown:roles.md'")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(title, "Doc roles.md");
}

#[sqlx::test(migrator = "recall::store::MIGRATOR")]
async fn identical_reingest_writes_nothing(pool: PgPool) {
    let store = ChunkStore::new(pool.clone());
    let embedder = RecordingEmbedder::new();
    let doc = page(&[("a", "alpha"), ("b", "beta")]);
    store.store(&doc, &embedder).await.unwrap();

    assert_eq!(store.store(&doc, &embedder).await.unwrap(), stored(0, 0));
    assert_eq!(count(&pool, "true").await, 2);
    assert_eq!(embedder.batches().len(), 1);
}

#[sqlx::test(migrator = "recall::store::MIGRATOR")]
async fn changed_chunk_supersedes_its_old_version(pool: PgPool) {
    let store = ChunkStore::new(pool.clone());
    let embedder = RecordingEmbedder::new();
    store
        .store(&page(&[("a", "alpha"), ("b", "beta")]), &embedder)
        .await
        .unwrap();

    let result = store
        .store(&page(&[("a", "alpha"), ("b", "beta, revised")]), &embedder)
        .await
        .unwrap();

    assert_eq!(result, stored(1, 1));
    assert_eq!(embedder.batches()[1], ["Doc roles.md\nB\n\nbeta, revised"]);
    assert_eq!(count(&pool, "true").await, 3);
    assert_eq!(
        count(&pool, "superseded_at IS NOT NULL AND text = 'beta'").await,
        1
    );
    assert_eq!(
        count(&pool, "superseded_at IS NULL AND text = 'beta, revised'").await,
        1
    );
}

#[sqlx::test(migrator = "recall::store::MIGRATOR")]
async fn removed_chunk_is_superseded(pool: PgPool) {
    let store = ChunkStore::new(pool.clone());
    let embedder = RecordingEmbedder::new();
    store
        .store(&page(&[("a", "alpha"), ("b", "beta")]), &embedder)
        .await
        .unwrap();

    let result = store
        .store(&page(&[("b", "beta")]), &embedder)
        .await
        .unwrap();

    assert_eq!(result, stored(0, 1));
    assert_eq!(
        current(&pool).await,
        [("markdown:roles.md#b".into(), 0, "fake-hash-embedder".into())]
    );
    assert_eq!(embedder.batches().len(), 1);
}

#[sqlx::test(migrator = "recall::store::MIGRATOR")]
async fn reverted_chunk_revives_its_earlier_version_without_embedding(pool: PgPool) {
    let store = ChunkStore::new(pool.clone());
    let embedder = RecordingEmbedder::new();
    store
        .store(&page(&[("a", "alpha")]), &embedder)
        .await
        .unwrap();
    store
        .store(&page(&[("a", "omega")]), &embedder)
        .await
        .unwrap();

    let result = store
        .store(&page(&[("a", "alpha")]), &embedder)
        .await
        .unwrap();

    assert_eq!(result, stored(1, 1));
    assert_eq!(embedder.batches().len(), 2);
    assert_eq!(count(&pool, "true").await, 2);
    assert_eq!(
        count(&pool, "superseded_at IS NULL AND text = 'alpha'").await,
        1
    );
}

#[sqlx::test(migrator = "recall::store::MIGRATOR")]
async fn reordered_chunks_keep_their_versions_and_take_new_ordinals(pool: PgPool) {
    let store = ChunkStore::new(pool.clone());
    let embedder = RecordingEmbedder::new();
    store
        .store(&page(&[("a", "alpha"), ("b", "beta")]), &embedder)
        .await
        .unwrap();

    let result = store
        .store(&page(&[("b", "beta"), ("a", "alpha")]), &embedder)
        .await
        .unwrap();

    assert_eq!(result, stored(0, 0));
    assert_eq!(count(&pool, "true").await, 2);
    assert_eq!(
        current(&pool).await,
        [
            ("markdown:roles.md#a".into(), 1, "fake-hash-embedder".into()),
            ("markdown:roles.md#b".into(), 0, "fake-hash-embedder".into()),
        ]
    );
}

#[sqlx::test(migrator = "recall::store::MIGRATOR")]
async fn duplicate_locators_are_rejected(pool: PgPool) {
    let store = ChunkStore::new(pool.clone());
    let doc = page(&[("a", "alpha"), ("a", "again")]);

    assert!(store.store(&doc, &HashEmbedder).await.is_err());
    assert_eq!(count(&pool, "true").await, 0);
}

#[sqlx::test(migrator = "recall::store::MIGRATOR")]
async fn embedding_model_mismatch_is_refused(pool: PgPool) {
    let sources = sources(vec![static_source(vec![page(&[("a", "alpha")])])]);
    ingest(&pool, &HashEmbedder, &sources, false).await.unwrap();

    let other = RecordingEmbedder::named(ModelRef::new("amazon.titan-embed-text-v2:0", "x"));
    let error = ingest(&pool, &other, &sources, false).await.unwrap_err();

    assert!(error.to_string().contains("--reembed"));
    let mismatch = error
        .narrow::<EmbeddingModelMismatch, _>()
        .expect("a model mismatch, not a database error");
    assert_eq!(mismatch.current, [HashEmbedder.model()]);
    assert!(other.batches().is_empty());
    assert_eq!(count(&pool, "true").await, 1);
}

#[sqlx::test(migrator = "recall::store::MIGRATOR")]
async fn reembed_replaces_every_chunk_with_the_configured_model(pool: PgPool) {
    let first = static_source(vec![
        page(&[("a", "alpha"), ("b", "beta")]),
        document(&markdown::NAME, "other.md", &[("c", "gamma")]),
    ]);
    ingest(&pool, &HashEmbedder, &sources(vec![first]), false)
        .await
        .unwrap();

    let subset = static_source(vec![page(&[("a", "alpha"), ("b", "beta")])]);
    let titan = RecordingEmbedder::named(ModelRef::new("titan", "1"));
    let reports = ingest(&pool, &titan, &sources(vec![subset]), true)
        .await
        .unwrap();

    assert!(reports[0].error.is_none());
    assert_eq!(reports[0].documents, 1);
    assert_eq!(reports[0].stored, stored(2, 3));
    assert_eq!(
        current(&pool).await,
        [
            ("markdown:roles.md#a".into(), 0, "titan".into()),
            ("markdown:roles.md#b".into(), 1, "titan".into()),
        ]
    );
    assert_eq!(
        ChunkStore::new(pool.clone())
            .current_embedding_models()
            .await
            .unwrap(),
        [ModelRef::new("titan", "1")]
    );
}

#[sqlx::test(migrator = "recall::store::MIGRATOR")]
async fn snapshot_supersedes_documents_missing_from_fetch(pool: PgPool) {
    let sources = sources(vec![static_source(vec![
        page(&[("a", "alpha")]),
        document(&markdown::NAME, "removed.md", &[("intro", "gone")]),
    ])]);
    ingest(&pool, &HashEmbedder, &sources, false).await.unwrap();
    assert_eq!(count(&pool, "superseded_at IS NULL").await, 2);

    let sources = self::sources(vec![static_source(vec![page(&[("a", "alpha")])])]);
    let reports = ingest(&pool, &HashEmbedder, &sources, false).await.unwrap();
    assert_eq!(reports[0].stored.superseded, 1);
    assert_eq!(count(&pool, "superseded_at IS NULL").await, 1);
    assert_eq!(
        count(
            &pool,
            "superseded_at IS NULL AND id LIKE 'markdown:removed.md%'"
        )
        .await,
        0
    );
}

/// `(chunks, avg_length)` of the corpus statistics, and the document frequency of each lexeme in `lexemes`.
async fn term_stats(pool: &PgPool, lexemes: &[&str]) -> ((i64, Option<f64>), Vec<Option<i32>>) {
    let corpus = sqlx::query_as("SELECT chunks, avg_length FROM fts_prose_corpus")
        .fetch_one(pool)
        .await
        .unwrap();
    let frequencies = sqlx::query_scalar(
        "SELECT t.chunks FROM unnest($1::text[]) WITH ORDINALITY l(lexeme, i)
         LEFT JOIN fts_prose_terms t USING (lexeme) ORDER BY i",
    )
    .bind(lexemes)
    .fetch_all(pool)
    .await
    .unwrap();
    (corpus, frequencies)
}

#[sqlx::test(migrator = "recall::store::MIGRATOR")]
async fn ingest_refreshes_term_stats_from_current_chunks(pool: PgPool) {
    assert_eq!(term_stats(&pool, &["role"]).await, ((0, None), vec![None]));

    // Headings are the uppercased locators, so `role` is in every chunk.
    let sources = self::sources(vec![static_source(vec![
        document(&markdown::NAME, "a.md", &[("role", "payroll payroll")]),
        document(&markdown::NAME, "b.md", &[("role", "role owners review")]),
    ])]);
    ingest(&pool, &HashEmbedder, &sources, false).await.unwrap();
    assert_eq!(
        term_stats(&pool, &["role", "payrol", "owner"]).await,
        ((2, Some(3.5)), vec![Some(2), Some(1), Some(1)])
    );

    let sources = self::sources(vec![static_source(vec![document(
        &markdown::NAME,
        "b.md",
        &[("role", "role owners review")],
    )])]);
    ingest(&pool, &HashEmbedder, &sources, false).await.unwrap();
    assert_eq!(
        term_stats(&pool, &["role", "payrol", "owner"]).await,
        ((1, Some(4.0)), vec![Some(1), None, Some(1)])
    );
}

#[sqlx::test(migrator = "recall::store::MIGRATOR")]
async fn one_failing_source_does_not_stop_the_others(pool: PgPool) {
    let sources = sources(vec![
        Box::new(FailingSource { name: wiki() }),
        static_source(vec![
            document(&markdown::NAME, "a.md", &[("intro", "question")]),
            document(&markdown::NAME, "b.md", &[("intro", "answer")]),
        ]),
    ]);

    let reports = ingest(&pool, &HashEmbedder, &sources, false).await.unwrap();

    assert_eq!(reports[0].source, wiki());
    assert!(format!("{:#?}", reports[0].error.as_ref().unwrap()).contains("source wiki is down"));
    assert_eq!(reports[1].source, markdown::NAME);
    assert!(reports[1].error.is_none());
    assert_eq!(reports[1].documents, 2);
    assert_eq!(reports[1].stored, stored(2, 0));
    assert_eq!(count(&pool, "superseded_at IS NULL").await, 2);
}

#[sqlx::test(migrator = "recall::store::MIGRATOR")]
async fn a_source_named_outside_the_crate_supersedes_only_its_own_documents(pool: PgPool) {
    let markdown = || static_source(vec![page(&[("a", "alpha")])]);
    let wiki_pages = |pages: &[&str]| {
        static_source(
            pages
                .iter()
                .map(|page| document(&wiki(), page, &[("intro", "beta")]))
                .collect(),
        )
    };
    ingest(
        &pool,
        &HashEmbedder,
        &sources(vec![markdown(), wiki_pages(&["home", "roles"])]),
        false,
    )
    .await
    .unwrap();

    let reports = ingest(
        &pool,
        &HashEmbedder,
        &sources(vec![wiki_pages(&["home"])]),
        false,
    )
    .await
    .unwrap();

    assert_eq!(reports[0].source, wiki());
    assert_eq!(reports[0].stored, stored(0, 1));
    assert_eq!(
        current(&pool)
            .await
            .into_iter()
            .map(|(id, _, _)| id)
            .collect::<Vec<_>>(),
        ["markdown:roles.md#a", "wiki:home#intro"]
    );
    let source: String = sqlx::query_scalar("SELECT source FROM documents WHERE id = 'wiki:home'")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(source, "wiki");
}

#[sqlx::test(migrator = "recall::store::MIGRATOR")]
async fn a_document_under_another_source_name_fails_its_source(pool: PgPool) {
    let mislabelled = Box::new(StaticSource {
        name: wiki(),
        documents: vec![page(&[("a", "alpha")])],
    });

    let reports = ingest(&pool, &HashEmbedder, &sources(vec![mislabelled]), false)
        .await
        .unwrap();

    let error = format!("{:#?}", reports[0].error.as_ref().unwrap());
    assert!(
        error.contains("wiki source returned document markdown:roles.md"),
        "{error}"
    );
    assert_eq!(count(&pool, "true").await, 0);
}
