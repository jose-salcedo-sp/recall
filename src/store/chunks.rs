//! Writes documents and chunk versions: insert new versions, supersede replaced and removed ones.

use std::collections::{HashMap, HashSet};

use eros::{Context, bail, ensure};
use pgvector::Vector;
use sqlx::PgPool;

use crate::{
    domain::{ChunkDraft, ChunkVersion, Document, DocumentId, SourceKind},
    llm::{EMBEDDING_DIM, EmbedPurpose, Embedder, ModelRef},
};

/// What storing one document changed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stored {
    /// Chunk versions that became current.
    pub new_versions: u64,
    /// Current versions replaced by a new version or no longer present in the document.
    pub superseded: u64,
}

impl std::ops::AddAssign for Stored {
    fn add_assign(&mut self, other: Self) {
        self.new_versions += other.new_versions;
        self.superseded += other.superseded;
    }
}

/// Writes chunk versions; cheap to clone (shares the pool).
#[derive(Clone)]
pub struct ChunkStore {
    pool: PgPool,
}

impl ChunkStore {
    /// Wraps a pool whose migrations have run.
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Makes the document's drafts its current chunks, embedding only drafts whose version is not already stored.
    ///
    /// # Errors
    ///
    /// When two drafts share a locator, embedding fails or returns the wrong count or dimension, or a query fails.
    pub async fn store(
        &self,
        document: &Document,
        embedder: &dyn Embedder,
    ) -> eros::Result<Stored> {
        let document_id = document.id();
        let model = embedder.model();

        let mut locators = HashSet::new();
        for draft in &document.chunks {
            ensure!(
                locators.insert(draft.locator.as_str()),
                "document {} has duplicate chunk locator {:?}",
                document_id.0,
                draft.locator
            );
        }

        let drafts: Vec<(String, ChunkVersion, &ChunkDraft)> = document
            .chunks
            .iter()
            .map(|draft| {
                let version = ChunkVersion::compute(document, draft, &model);
                (document.chunk_id(draft).0, version, draft)
            })
            .collect();

        let current: HashMap<String, (String, i32)> = sqlx::query_as::<_, (String, String, i32)>(
            "SELECT id, version, ordinal FROM chunks WHERE document_id = $1 AND superseded_at IS NULL",
        )
        .bind(&document_id.0)
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|(id, version, ordinal)| (id, (version, ordinal)))
        .collect();

        let mut moved = Vec::new();
        let mut changed = Vec::new();
        for (ordinal, (id, version, draft)) in drafts.iter().enumerate() {
            let ordinal = i32::try_from(ordinal)?;
            match current.get(id) {
                Some((v, o)) if *v == version.0 => {
                    if *o != ordinal {
                        moved.push((id, version, ordinal));
                    }
                }
                _ => changed.push((id, version, ordinal, *draft)),
            }
        }
        let kept: HashSet<(&str, &str)> = drafts
            .iter()
            .map(|(id, version, _)| (id.as_str(), version.0.as_str()))
            .collect();
        let stale: Vec<&str> = current
            .iter()
            .filter(|(id, (version, _))| !kept.contains(&(id.as_str(), version.as_str())))
            .map(|(id, _)| id.as_str())
            .collect();

        // A draft can return to a version stored earlier (an edit reverted); that row is revived, not re-embedded.
        let (ids, versions): (Vec<&str>, Vec<&str>) = changed
            .iter()
            .map(|(id, version, _, _)| (id.as_str(), version.0.as_str()))
            .unzip();
        let stored: HashSet<(String, String)> = sqlx::query_as::<_, (String, String)>(
            "SELECT id, version FROM chunks
             WHERE (id, version) IN (SELECT * FROM UNNEST($1::text[], $2::text[]))",
        )
        .bind(&ids)
        .bind(&versions)
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .collect();
        let (revived, fresh): (Vec<_>, Vec<_>) = changed
            .into_iter()
            .partition(|(id, version, _, _)| stored.contains(&((*id).clone(), version.0.clone())));

        let embeddings = if fresh.is_empty() {
            Vec::new()
        } else {
            let texts: Vec<String> = fresh
                .iter()
                .map(|(_, _, _, draft)| embedding_input(document, draft))
                .collect();
            let embeddings = embedder
                .embed(&texts, EmbedPurpose::Document)
                .await
                .with_context(|| format!("embedding chunks of {}", document_id.0))?;
            ensure!(
                embeddings.len() == texts.len(),
                "embedder returned {} vectors for {} texts",
                embeddings.len(),
                texts.len()
            );
            if let Some(bad) = embeddings.iter().find(|e| e.len() != EMBEDDING_DIM) {
                bail!(
                    "embedder returned a {}-dimensional vector, expected {EMBEDDING_DIM}",
                    bad.len()
                );
            }
            embeddings
        };

        let mut tx = self.pool.begin().await?;
        sqlx::query(
            "INSERT INTO documents (id, source, external_id, title, url, source_updated_at)
             VALUES ($1, $2, $3, $4, $5, $6)
             ON CONFLICT (id) DO UPDATE SET
                 title = EXCLUDED.title,
                 url = EXCLUDED.url,
                 source_updated_at = EXCLUDED.source_updated_at,
                 last_ingested_at = now()",
        )
        .bind(&document_id.0)
        .bind(document.source.as_str())
        .bind(&document.external_id)
        .bind(&document.title)
        .bind(&document.url)
        .bind(document.updated_at)
        .execute(&mut *tx)
        .await?;

        let superseded = sqlx::query(
            "UPDATE chunks SET superseded_at = now()
             WHERE document_id = $1 AND superseded_at IS NULL AND id = ANY($2)",
        )
        .bind(&document_id.0)
        .bind(&stale)
        .execute(&mut *tx)
        .await?
        .rows_affected();

        for (id, version, ordinal) in &moved {
            sqlx::query("UPDATE chunks SET ordinal = $3 WHERE id = $1 AND version = $2")
                .bind(id)
                .bind(&version.0)
                .bind(ordinal)
                .execute(&mut *tx)
                .await?;
        }

        for (id, version, ordinal, _) in &revived {
            sqlx::query(
                "UPDATE chunks SET superseded_at = NULL, ordinal = $3 WHERE id = $1 AND version = $2",
            )
            .bind(id)
            .bind(&version.0)
            .bind(ordinal)
            .execute(&mut *tx)
            .await?;
        }

        for ((id, version, ordinal, draft), embedding) in fresh.iter().zip(embeddings) {
            sqlx::query(
                "INSERT INTO chunks (id, version, document_id, ordinal, heading, url, text, authoritative,
                                     acl, customer, sensitivity, embedding, embedding_model,
                                     embedding_model_version)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)",
            )
            .bind(id)
            .bind(&version.0)
            .bind(&document_id.0)
            .bind(ordinal)
            .bind(&draft.heading)
            .bind(&draft.url)
            .bind(&draft.text)
            .bind(draft.authoritative)
            .bind(&document.labels.acl)
            .bind(&document.labels.customer)
            .bind(document.labels.sensitivity.as_str())
            .bind(Vector::from(embedding))
            .bind(&model.id)
            .bind(&model.version)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;

        Ok(Stored {
            new_versions: (revived.len() + fresh.len()) as u64,
            superseded,
        })
    }

    /// Distinct embedding models among current chunks, sorted by id then version.
    ///
    /// # Errors
    ///
    /// When the query fails.
    pub async fn current_embedding_models(&self) -> eros::Result<Vec<ModelRef>, (sqlx::Error,)> {
        Ok(sqlx::query_as::<_, (String, String)>(
            "SELECT DISTINCT embedding_model, embedding_model_version FROM chunks
             WHERE superseded_at IS NULL ORDER BY 1, 2",
        )
        .fetch_all(&self.pool)
        .await?
        .into_iter()
        .map(|(id, version)| ModelRef::new(id, version))
        .collect())
    }

    /// Supersedes every current chunk embedded with a model other than `model`. Returns how many.
    ///
    /// # Errors
    ///
    /// When the query fails.
    pub async fn supersede_other_models(
        &self,
        model: &ModelRef,
    ) -> eros::Result<u64, (sqlx::Error,)> {
        Ok(sqlx::query(
            "UPDATE chunks SET superseded_at = now()
             WHERE superseded_at IS NULL
               AND (embedding_model, embedding_model_version) IS DISTINCT FROM ($1, $2)",
        )
        .bind(&model.id)
        .bind(&model.version)
        .execute(&self.pool)
        .await?
        .rows_affected())
    }

    /// Recomputes the full-text statistics that BM25 ranking reads (document frequency per lexeme, chunk count and
    /// average length) from the current chunks, in one transaction.
    ///
    /// # Errors
    ///
    /// When the query fails.
    pub async fn refresh_term_stats(&self) -> eros::Result<(), (sqlx::Error,)> {
        sqlx::query("SELECT refresh_fts_prose_stats()")
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Supersedes current chunks of documents for `source` whose ids are not in `present`. Returns how many.
    ///
    /// # Errors
    ///
    /// When the query fails.
    pub async fn supersede_missing_documents(
        &self,
        source: SourceKind,
        present: &[DocumentId],
    ) -> eros::Result<u64, (sqlx::Error,)> {
        let ids: Vec<&str> = present.iter().map(|d| d.0.as_str()).collect();
        Ok(sqlx::query(
            "UPDATE chunks SET superseded_at = now()
             WHERE superseded_at IS NULL
               AND document_id IN (
                 SELECT id FROM documents WHERE source = $1 AND NOT (id = ANY($2))
               )",
        )
        .bind(source.as_str())
        .bind(&ids)
        .execute(&self.pool)
        .await?
        .rows_affected())
    }
}

/// The document title and section heading give a section's text the context it lacks on its own.
fn embedding_input(document: &Document, draft: &ChunkDraft) -> String {
    match &draft.heading {
        Some(heading) => format!("{}\n{heading}\n\n{}", document.title, draft.text),
        None => format!("{}\n\n{}", document.title, draft.text),
    }
}
