//! The rules every [`Source`] keeps, as one async check per rule over `&dyn Source`. [`source_contract!`] turns them
//! into tests; each connector, the fakes included, calls it once with a source holding at least one document.
//!
//! ```text
//! recall::source_contract!(#[tokio::test] () => MyConnector::new(fixture_config()));
//! ```
//!
//! [`source_contract!`]: crate::source_contract

use std::collections::HashSet;

use super::Source;
use crate::domain::Document;

/// The source's snapshot, which must hold at least one document for the checks to mean anything.
async fn snapshot(source: &dyn Source) -> eros::Result<Vec<Document>> {
    let documents = source.fetch_all().await?;
    assert!(
        !documents.is_empty(),
        "the contract needs a source holding at least one document"
    );
    Ok(documents)
}

/// Every document carries [`Source::name`], so ingestion accepts it and its snapshot supersedes the right rows.
///
/// # Errors
///
/// When the fetch fails.
pub async fn every_document_carries_the_source_name(source: &dyn Source) -> eros::Result<()> {
    for document in snapshot(source).await? {
        assert_eq!(&document.source, source.name(), "{}", document.id().0);
    }
    Ok(())
}

/// No two documents in one snapshot share an external id, so neither overwrites the other.
///
/// # Errors
///
/// When the fetch fails.
pub async fn external_ids_are_unique_and_non_empty(source: &dyn Source) -> eros::Result<()> {
    let documents = snapshot(source).await?;
    let mut ids = HashSet::new();
    for document in &documents {
        assert!(!document.external_id.is_empty(), "an empty external id");
        assert!(
            ids.insert(document.id()),
            "{} is listed twice",
            document.id().0
        );
    }
    Ok(())
}

/// Locators are unique within a document, and every chunk has text to embed.
///
/// # Errors
///
/// When the fetch fails.
pub async fn chunks_have_unique_locators_and_text(source: &dyn Source) -> eros::Result<()> {
    for document in snapshot(source).await? {
        let mut locators = HashSet::new();
        for draft in &document.chunks {
            assert!(
                locators.insert(draft.locator.as_str()),
                "{} repeats locator {:?}",
                document.id().0,
                draft.locator
            );
            assert!(
                !draft.text.trim().is_empty(),
                "{} has an empty chunk {:?}",
                document.id().0,
                draft.locator
            );
        }
    }
    Ok(())
}

/// Two fetches of an unchanged source return the same documents, so re-ingesting it stores no new version. Order
/// between documents is free; order within a document is not, since it is the chunks' ordinal.
///
/// # Errors
///
/// When either fetch fails.
pub async fn an_unchanged_source_returns_the_same_snapshot(
    source: &dyn Source,
) -> eros::Result<()> {
    let by_id = |mut documents: Vec<Document>| {
        documents.sort_by_key(Document::id);
        documents
    };
    let first = by_id(snapshot(source).await?);
    let second = by_id(snapshot(source).await?);
    assert_eq!(first, second);
    Ok(())
}

/// One test per [`crate::ingest::contract`] check, for the source `$make` builds. The caller gives the test
/// attribute and any arguments the harness passes, e.g. `#[tokio::test] () => StaticSource { .. }`.
#[macro_export]
macro_rules! source_contract {
    (#[$attr:meta] ($($arg:ident: $ty:ty),*) => $make:expr) => {
        #[$attr]
        async fn every_document_carries_the_source_name($($arg: $ty),*) -> eros::Result<()> {
            $crate::ingest::contract::every_document_carries_the_source_name(&$make).await
        }

        #[$attr]
        async fn external_ids_are_unique_and_non_empty($($arg: $ty),*) -> eros::Result<()> {
            $crate::ingest::contract::external_ids_are_unique_and_non_empty(&$make).await
        }

        #[$attr]
        async fn chunks_have_unique_locators_and_text($($arg: $ty),*) -> eros::Result<()> {
            $crate::ingest::contract::chunks_have_unique_locators_and_text(&$make).await
        }

        #[$attr]
        async fn an_unchanged_source_returns_the_same_snapshot($($arg: $ty),*) -> eros::Result<()> {
            $crate::ingest::contract::an_unchanged_source_returns_the_same_snapshot(&$make).await
        }
    };
}
