//! The `ChunkStore` rules, on Postgres and on the in-memory store. Each check observes the store only through the
//! port: what `store` reports, what it asked the embedder for, and what the supersede calls count.

use chrono::{TimeZone, Utc};
use recall::{
    domain::{ChunkDraft, Document, DocumentId, Labels, Sensitivity, SourceName},
    llm::{Embedder, ModelRef, fakes::RecordingEmbedder},
    store::chunks::{ChunkStore, Stored},
};

fn source(name: &'static str) -> SourceName {
    SourceName::from_static(name)
}

fn document(source: &SourceName, external_id: &str, drafts: &[(&str, &str)]) -> Document {
    Document {
        source: source.clone(),
        external_id: external_id.into(),
        title: format!("Doc {external_id}"),
        url: format!("https://example.test/{external_id}"),
        updated_at: Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap(),
        labels: Labels {
            acl: recall::access::Acl::anyone(),
            tenant: None,
            sensitivity: Sensitivity::Internal,
        },
        chunks: drafts
            .iter()
            .map(|(locator, text)| ChunkDraft {
                locator: (*locator).into(),
                heading: Some((*locator).into()),
                url: format!("https://example.test/{external_id}#{locator}"),
                text: (*text).into(),
                authoritative: true,
            })
            .collect(),
    }
}

fn page(drafts: &[(&str, &str)]) -> Document {
    document(&source("markdown"), "roles.md", drafts)
}

fn stored(new_versions: u64, superseded: u64) -> Stored {
    Stored {
        new_versions,
        superseded,
    }
}

/// Texts embedded per batch, without the title and heading prefix.
fn embedded(embedder: &RecordingEmbedder) -> Vec<usize> {
    embedder.batches().iter().map(Vec::len).collect()
}

pub async fn the_first_store_embeds_and_inserts_every_draft(
    store: &dyn ChunkStore,
) -> eros::Result<()> {
    let embedder = RecordingEmbedder::default();

    let result = store
        .store(&page(&[("a", "alpha"), ("b", "beta")]), &embedder)
        .await?;

    assert_eq!(result, stored(2, 0));
    assert_eq!(embedded(&embedder), [2]);
    assert_eq!(store.current_embedding_models().await?, [embedder.model()]);
    Ok(())
}

pub async fn an_unchanged_document_stores_and_embeds_nothing(
    store: &dyn ChunkStore,
) -> eros::Result<()> {
    let embedder = RecordingEmbedder::default();
    let doc = page(&[("a", "alpha"), ("b", "beta")]);
    store.store(&doc, &embedder).await?;

    assert_eq!(store.store(&doc, &embedder).await?, stored(0, 0));
    assert_eq!(embedded(&embedder), [2]);
    Ok(())
}

pub async fn an_edit_supersedes_and_embeds_only_the_edited_draft(
    store: &dyn ChunkStore,
) -> eros::Result<()> {
    let embedder = RecordingEmbedder::default();
    store
        .store(&page(&[("a", "alpha"), ("b", "beta")]), &embedder)
        .await?;

    let result = store
        .store(&page(&[("a", "alpha"), ("b", "beta, edited")]), &embedder)
        .await?;

    assert_eq!(result, stored(1, 1));
    assert_eq!(embedded(&embedder), [2, 1]);
    assert!(embedder.batches()[1][0].ends_with("beta, edited"));
    Ok(())
}

pub async fn a_reverted_edit_is_revived_without_embedding(
    store: &dyn ChunkStore,
) -> eros::Result<()> {
    let embedder = RecordingEmbedder::default();
    for text in ["first", "second"] {
        store.store(&page(&[("a", text)]), &embedder).await?;
    }

    let result = store.store(&page(&[("a", "first")]), &embedder).await?;

    assert_eq!(result, stored(1, 1));
    assert_eq!(embedded(&embedder), [1, 1]);
    Ok(())
}

pub async fn a_removed_draft_is_superseded(store: &dyn ChunkStore) -> eros::Result<()> {
    let embedder = RecordingEmbedder::default();
    store
        .store(&page(&[("a", "alpha"), ("b", "beta")]), &embedder)
        .await?;

    assert_eq!(
        store.store(&page(&[("a", "alpha")]), &embedder).await?,
        stored(0, 1)
    );
    Ok(())
}

pub async fn reordering_drafts_stores_no_new_version(store: &dyn ChunkStore) -> eros::Result<()> {
    let embedder = RecordingEmbedder::default();
    store
        .store(&page(&[("a", "alpha"), ("b", "beta")]), &embedder)
        .await?;

    assert_eq!(
        store
            .store(&page(&[("b", "beta"), ("a", "alpha")]), &embedder)
            .await?,
        stored(0, 0)
    );
    assert_eq!(embedded(&embedder), [2]);
    Ok(())
}

pub async fn relabelling_a_document_makes_new_versions(store: &dyn ChunkStore) -> eros::Result<()> {
    let embedder = RecordingEmbedder::default();
    let mut doc = page(&[("a", "alpha")]);
    store.store(&doc, &embedder).await?;

    doc.labels.sensitivity = Sensitivity::Confidential;

    assert_eq!(store.store(&doc, &embedder).await?, stored(1, 1));
    Ok(())
}

pub async fn a_rejected_document_stores_nothing(store: &dyn ChunkStore) -> eros::Result<()> {
    let embedder = RecordingEmbedder::default();

    let duplicate = store
        .store(&page(&[("a", "alpha"), ("a", "again")]), &embedder)
        .await;
    let unembedded = store
        .store(&page(&[("b", "beta")]), &recall::llm::fakes::FailingModel)
        .await;

    assert!(duplicate.is_err());
    assert!(unembedded.is_err());
    assert!(store.current_embedding_models().await?.is_empty());
    Ok(())
}

pub async fn models_are_listed_sorted_and_others_superseded(
    store: &dyn ChunkStore,
) -> eros::Result<()> {
    let zeta = RecordingEmbedder::named(ModelRef::new("zeta", "1"));
    let alpha = RecordingEmbedder::named(ModelRef::new("alpha", "2"));
    let markdown = source("markdown");
    store
        .store(
            &document(&markdown, "z.md", &[("a", "z1"), ("b", "z2")]),
            &zeta,
        )
        .await?;
    store
        .store(&document(&markdown, "a.md", &[("a", "a1")]), &alpha)
        .await?;

    assert_eq!(
        store.current_embedding_models().await?,
        [ModelRef::new("alpha", "2"), ModelRef::new("zeta", "1")]
    );
    assert_eq!(
        store
            .supersede_other_models(&ModelRef::new("alpha", "2"))
            .await?,
        2
    );
    assert_eq!(
        store.current_embedding_models().await?,
        [ModelRef::new("alpha", "2")]
    );
    assert_eq!(
        store
            .supersede_other_models(&ModelRef::new("alpha", "2"))
            .await?,
        0
    );
    Ok(())
}

pub async fn missing_documents_are_superseded_only_for_their_source(
    store: &dyn ChunkStore,
) -> eros::Result<()> {
    let embedder = RecordingEmbedder::default();
    let (markdown, wiki) = (source("markdown"), source("wiki"));
    for doc in [
        document(&markdown, "kept.md", &[("a", "kept")]),
        document(&markdown, "gone.md", &[("a", "gone"), ("b", "gone too")]),
        document(&wiki, "home", &[("a", "home")]),
    ] {
        store.store(&doc, &embedder).await?;
    }
    let kept = [DocumentId::new(&markdown, "kept.md")];

    assert_eq!(
        store.supersede_missing_documents(&markdown, &kept).await?,
        2
    );
    assert_eq!(
        store.supersede_missing_documents(&markdown, &kept).await?,
        0
    );
    assert_eq!(store.supersede_missing_documents(&wiki, &[]).await?, 1);
    store.refresh_term_stats().await?;
    Ok(())
}

macro_rules! chunk_store_contract {
    (#[$attr:meta] $args:tt => $make:expr) => {
        chunk_store_contract!(@cases #[$attr] $args => $make;
            the_first_store_embeds_and_inserts_every_draft,
            an_unchanged_document_stores_and_embeds_nothing,
            an_edit_supersedes_and_embeds_only_the_edited_draft,
            a_reverted_edit_is_revived_without_embedding,
            a_removed_draft_is_superseded,
            reordering_drafts_stores_no_new_version,
            relabelling_a_document_makes_new_versions,
            a_rejected_document_stores_nothing,
            models_are_listed_sorted_and_others_superseded,
            missing_documents_are_superseded_only_for_their_source,
        );
    };
    (@cases #[$attr:meta] $args:tt => $make:expr; $($case:ident,)*) => {
        $(
            #[$attr]
            async fn $case $args -> eros::Result<()> {
                crate::chunk_store::$case(&$make).await
            }
        )*
    };
}

mod memory {
    use recall::store::fakes::MemoryChunkStore;

    chunk_store_contract!(#[tokio::test] () => MemoryChunkStore::default());
}

mod postgres {
    use recall::store::chunks::PgChunkStore;

    chunk_store_contract!(
        #[sqlx::test(migrator = "recall::store::MIGRATOR")] (pool: sqlx::PgPool) => PgChunkStore::new(pool)
    );
}

#[tokio::test]
async fn the_failing_store_fails_every_call() {
    use recall::store::fakes::FailingChunkStore;
    let embedder = RecordingEmbedder::default();
    let store = FailingChunkStore;
    assert!(
        store
            .store(&page(&[("a", "alpha")]), &embedder)
            .await
            .is_err()
    );
    assert!(
        store
            .supersede_missing_documents(&source("markdown"), &[])
            .await
            .is_err()
    );
    assert!(store.current_embedding_models().await.is_err());
    assert!(
        store
            .supersede_other_models(&ModelRef::new("m", "1"))
            .await
            .is_err()
    );
    assert!(store.refresh_term_stats().await.is_err());
}
