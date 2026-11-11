//! `recall::ingest::ingest` against Postgres: versioning, superseding, and the embedding-model check.

#![allow(
    clippy::unwrap_used,
    reason = "integration-test helpers; clippy exempts only #[test] functions"
)]

use chrono::{DateTime, TimeZone, Utc};
use recall::{
    access::{Acl, ConnectorLabels, fakes::FailingLabeler, rules::AccessConfig},
    domain::{ChunkDraft, Document, Labels, Sensitivity, SourceName},
    ingest::{
        EmbeddingModelMismatch, Source, Sources,
        fakes::{FailingSource, StaticSource},
        ingest, markdown,
    },
    llm::{
        Embedder, ModelRef,
        fakes::{FailingModel, HashEmbedder, RecordingEmbedder},
    },
    store::{
        chunks::{ChunkStore, PgChunkStore, Stored},
        fakes::{FailingChunkStore, MemoryChunkStore},
    },
};
use sqlx::PgPool;

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
            acl: Acl::only(["role:arm".parse().unwrap()]),
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
    let store = PgChunkStore::new(pool.clone());
    let embedder = RecordingEmbedder::default();

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
    let store = PgChunkStore::new(pool.clone());
    let embedder = RecordingEmbedder::default();
    let doc = page(&[("a", "alpha"), ("b", "beta")]);
    store.store(&doc, &embedder).await.unwrap();

    assert_eq!(store.store(&doc, &embedder).await.unwrap(), stored(0, 0));
    assert_eq!(count(&pool, "true").await, 2);
    assert_eq!(embedder.batches().len(), 1);
}

#[sqlx::test(migrator = "recall::store::MIGRATOR")]
async fn changed_chunk_supersedes_its_old_version(pool: PgPool) {
    let store = PgChunkStore::new(pool.clone());
    let embedder = RecordingEmbedder::default();
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
    let store = PgChunkStore::new(pool.clone());
    let embedder = RecordingEmbedder::default();
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
    let store = PgChunkStore::new(pool.clone());
    let embedder = RecordingEmbedder::default();
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
    let store = PgChunkStore::new(pool.clone());
    let embedder = RecordingEmbedder::default();
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
    let store = PgChunkStore::new(pool.clone());
    let doc = page(&[("a", "alpha"), ("a", "again")]);

    assert!(store.store(&doc, &HashEmbedder).await.is_err());
    assert_eq!(count(&pool, "true").await, 0);
}

#[sqlx::test(migrator = "recall::store::MIGRATOR")]
async fn embedding_model_mismatch_is_refused(pool: PgPool) {
    let sources = sources(vec![static_source(vec![page(&[("a", "alpha")])])]);
    ingest(
        &PgChunkStore::new(pool.clone()),
        &HashEmbedder,
        &ConnectorLabels,
        &sources,
        false,
    )
    .await
    .unwrap();

    let other = RecordingEmbedder::named(ModelRef::new("amazon.titan-embed-text-v2:0", "x"));
    let error = ingest(
        &PgChunkStore::new(pool.clone()),
        &other,
        &ConnectorLabels,
        &sources,
        false,
    )
    .await
    .unwrap_err();

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
    ingest(
        &PgChunkStore::new(pool.clone()),
        &HashEmbedder,
        &ConnectorLabels,
        &sources(vec![first]),
        false,
    )
    .await
    .unwrap();

    let subset = static_source(vec![page(&[("a", "alpha"), ("b", "beta")])]);
    let titan = RecordingEmbedder::named(ModelRef::new("titan", "1"));
    let reports = ingest(
        &PgChunkStore::new(pool.clone()),
        &titan,
        &ConnectorLabels,
        &sources(vec![subset]),
        true,
    )
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
        PgChunkStore::new(pool.clone())
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
    ingest(
        &PgChunkStore::new(pool.clone()),
        &HashEmbedder,
        &ConnectorLabels,
        &sources,
        false,
    )
    .await
    .unwrap();
    assert_eq!(count(&pool, "superseded_at IS NULL").await, 2);

    let sources = self::sources(vec![static_source(vec![page(&[("a", "alpha")])])]);
    let reports = ingest(
        &PgChunkStore::new(pool.clone()),
        &HashEmbedder,
        &ConnectorLabels,
        &sources,
        false,
    )
    .await
    .unwrap();
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
    ingest(
        &PgChunkStore::new(pool.clone()),
        &HashEmbedder,
        &ConnectorLabels,
        &sources,
        false,
    )
    .await
    .unwrap();
    assert_eq!(
        term_stats(&pool, &["role", "payrol", "owner"]).await,
        ((2, Some(3.5)), vec![Some(2), Some(1), Some(1)])
    );

    let sources = self::sources(vec![static_source(vec![document(
        &markdown::NAME,
        "b.md",
        &[("role", "role owners review")],
    )])]);
    ingest(
        &PgChunkStore::new(pool.clone()),
        &HashEmbedder,
        &ConnectorLabels,
        &sources,
        false,
    )
    .await
    .unwrap();
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

    let reports = ingest(
        &PgChunkStore::new(pool.clone()),
        &HashEmbedder,
        &ConnectorLabels,
        &sources,
        false,
    )
    .await
    .unwrap();

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
        &PgChunkStore::new(pool.clone()),
        &HashEmbedder,
        &ConnectorLabels,
        &sources(vec![markdown(), wiki_pages(&["home", "roles"])]),
        false,
    )
    .await
    .unwrap();

    let reports = ingest(
        &PgChunkStore::new(pool.clone()),
        &HashEmbedder,
        &ConnectorLabels,
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

    let reports = ingest(
        &PgChunkStore::new(pool.clone()),
        &HashEmbedder,
        &ConnectorLabels,
        &sources(vec![mislabelled]),
        false,
    )
    .await
    .unwrap();

    let error = format!("{:#?}", reports[0].error.as_ref().unwrap());
    assert!(
        error.contains("wiki source returned document markdown:roles.md"),
        "{error}"
    );
    assert_eq!(count(&pool, "true").await, 0);
}

/// `(id, acl, sensitivity)` of current chunks, ordered by id.
async fn current_labels(pool: &PgPool) -> Vec<(String, Vec<String>, String)> {
    sqlx::query_as(
        "SELECT id, acl, sensitivity FROM chunks WHERE superseded_at IS NULL ORDER BY id",
    )
    .fetch_all(pool)
    .await
    .unwrap()
}

#[sqlx::test(migrator = "recall::store::MIGRATOR")]
async fn label_rules_set_the_stored_labels_and_relabelling_makes_a_new_version(pool: PgPool) {
    let documents = || {
        static_source(vec![
            document(&markdown::NAME, "hr/pay.md", &[("bands", "Salary bands.")]),
            document(&markdown::NAME, "eng/roles.md", &[("intro", "Roles.")]),
        ])
    };
    let rules = AccessConfig::parse(
        r#"{ "roles": {}, "labels": [
            { "source": "markdown", "path": "hr/", "acl": ["role:arm"], "sensitivity": "confidential" }
        ] }"#,
    )
    .unwrap()
    .labels()
    .clone();
    ingest(
        &PgChunkStore::new(pool.clone()),
        &HashEmbedder,
        &ConnectorLabels,
        &sources(vec![documents()]),
        false,
    )
    .await
    .unwrap();

    let reports = ingest(
        &PgChunkStore::new(pool.clone()),
        &HashEmbedder,
        &rules,
        &sources(vec![documents()]),
        false,
    )
    .await
    .unwrap();

    assert_eq!(reports[0].stored, stored(1, 1));
    assert_eq!(
        current_labels(&pool).await,
        [
            (
                "markdown:eng/roles.md#intro".into(),
                vec!["role:arm".into()],
                "internal".into()
            ),
            (
                "markdown:hr/pay.md#bands".into(),
                vec!["role:arm".into()],
                "confidential".into()
            ),
        ]
    );
}

#[sqlx::test(migrator = "recall::store::MIGRATOR")]
async fn a_document_that_cannot_be_labelled_is_not_stored(pool: PgPool) {
    let sources = sources(vec![static_source(vec![page(&[("a", "alpha")])])]);

    let reports = ingest(
        &PgChunkStore::new(pool.clone()),
        &HashEmbedder,
        &FailingLabeler,
        &sources,
        false,
    )
    .await
    .unwrap();

    let error = reports[0].error.as_ref().expect("the source failed");
    assert!(
        format!("{error:#?}").contains("labels for markdown:roles.md"),
        "{error:#?}"
    );
    assert_eq!(count(&pool, "true").await, 0);
}

#[tokio::test]
async fn a_failing_embedder_fails_only_its_source() {
    let store = MemoryChunkStore::default();
    let sources = sources(vec![static_source(vec![page(&[("a", "alpha")])])]);

    let reports = ingest(&store, &FailingModel, &ConnectorLabels, &sources, false)
        .await
        .unwrap();

    let error = reports[0].error.as_ref().expect("the source failed");
    assert!(
        format!("{error:#?}").contains("connection refused"),
        "{error:#?}"
    );
    assert!(store.current().is_empty());
}

#[tokio::test]
async fn a_store_that_cannot_answer_fails_ingestion_before_any_source() {
    let sources = sources(vec![static_source(vec![page(&[("a", "alpha")])])]);

    let error = ingest(
        &FailingChunkStore,
        &HashEmbedder,
        &ConnectorLabels,
        &sources,
        false,
    )
    .await
    .unwrap_err();

    assert!(
        error.to_string().contains("the chunk store failed"),
        "{error}"
    );
}

#[tokio::test]
async fn ingestion_into_memory_is_readable_by_a_retriever() {
    use recall::{
        access::fakes::unrestricted,
        retrieval::{Retriever, fakes::KeywordRetriever},
    };
    let store = MemoryChunkStore::default();
    let mut roles = page(&[
        ("roles", "ROLE_USERS maps users to roles."),
        ("lunch", "Lunch is at noon."),
    ]);
    roles.labels.acl = Acl::anyone();
    let sources = sources(vec![static_source(vec![roles])]);
    ingest(&store, &HashEmbedder, &ConnectorLabels, &sources, false)
        .await
        .unwrap();

    let retriever = KeywordRetriever {
        chunks: store.current(),
    };
    let hits = retriever
        .retrieve("Which table maps users to roles?", unrestricted().chunks())
        .await
        .unwrap();

    assert_eq!(hits[0].chunk.id.0, "markdown:roles.md#roles");
}
