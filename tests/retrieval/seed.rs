//! Seeds `documents` and `chunks` with plain SQL, independent of the ingestion code.

use std::collections::HashSet;

use drew::{
    domain::{ChunkId, DocumentId, SourceKind},
    llm::{EMBEDDING_DIM, EmbedPurpose, Embedder, ModelRef, fakes::HashEmbedder},
    retrieval::Retrieved,
};
use pgvector::Vector;
use sqlx::PgPool;

pub async fn document(
    pool: &PgPool,
    source: SourceKind,
    external_id: &str,
    title: &str,
) -> DocumentId {
    let id = DocumentId::new(source, external_id);
    sqlx::query(
        "INSERT INTO documents (id, source, external_id, title, url, source_updated_at)
         VALUES ($1, $2, $3, $4, $5, now())",
    )
    .bind(&id.0)
    .bind(source.as_str())
    .bind(external_id)
    .bind(title)
    .bind(format!("https://example.test/{}", id.0))
    .execute(pool)
    .await
    .expect("insert document");
    id
}

pub async fn embed(text: &str) -> Vec<f32> {
    HashEmbedder
        .embed(&[text.to_owned()], EmbedPurpose::Document)
        .await
        .expect("hash embedding")
        .remove(0)
}

/// A unit vector with cosine similarity `cosine` to the unit vector `target`.
pub fn similar_to(target: &[f32], cosine: f32) -> Vec<f32> {
    let orthogonal = target
        .iter()
        .position(|x| *x == 0.0)
        .expect("a sparse target");
    let mut vector: Vec<f32> = target.iter().map(|x| x * cosine).collect();
    vector[orthogonal] = (1.0 - cosine * cosine).sqrt();
    vector
}

pub struct SeedChunk {
    document: DocumentId,
    locator: String,
    ordinal: i32,
    heading: Option<String>,
    text: String,
    embedding: Option<Vec<f32>>,
    model: ModelRef,
    version: String,
    superseded: bool,
}

impl SeedChunk {
    /// A current chunk embedded with `HashEmbedder` over its text.
    pub fn new(document: &DocumentId, locator: &str, ordinal: i32, text: &str) -> Self {
        Self {
            document: document.clone(),
            locator: locator.into(),
            ordinal,
            heading: None,
            text: text.into(),
            embedding: None,
            model: HashEmbedder.model(),
            version: "v1".into(),
            superseded: false,
        }
    }

    pub fn heading(mut self, heading: &str) -> Self {
        self.heading = Some(heading.into());
        self
    }

    pub fn embedding(mut self, embedding: Vec<f32>) -> Self {
        assert_eq!(embedding.len(), EMBEDDING_DIM);
        self.embedding = Some(embedding);
        self
    }

    pub fn model(mut self, model: ModelRef) -> Self {
        self.model = model;
        self
    }

    pub fn version(mut self, version: &str) -> Self {
        self.version = version.into();
        self
    }

    pub fn superseded(mut self) -> Self {
        self.superseded = true;
        self
    }

    pub async fn insert(self, pool: &PgPool) -> ChunkId {
        let id = ChunkId::new(&self.document, &self.locator);
        let embedding = match self.embedding {
            Some(embedding) => embedding,
            None => embed(&self.text).await,
        };
        sqlx::query(
            "INSERT INTO chunks (id, version, document_id, ordinal, heading, url, text, authoritative, acl,
                                 customer, sensitivity, embedding, embedding_model, embedding_model_version,
                                 superseded_at)
             VALUES ($1, $2, $3, $4, $5, $6, $7, false, '{group:arm}', NULL, 'internal', $8, $9, $10,
                     CASE WHEN $11 THEN now() END)",
        )
        .bind(&id.0)
        .bind(&self.version)
        .bind(&self.document.0)
        .bind(self.ordinal)
        .bind(&self.heading)
        .bind(format!("https://example.test/{}", id.0))
        .bind(&self.text)
        .bind(Vector::from(embedding))
        .bind(&self.model.id)
        .bind(&self.model.version)
        .bind(self.superseded)
        .execute(pool)
        .await
        .expect("insert chunk");
        id
    }
}

/// Fraction of `relevant` chunks found in the first `k` results.
pub fn recall_at_k(retrieved: &[Retrieved], relevant: &[ChunkId], k: usize) -> f64 {
    let top: HashSet<&ChunkId> = retrieved.iter().take(k).map(|r| &r.chunk.id).collect();
    let found = relevant.iter().filter(|id| top.contains(id)).count();
    found as f64 / relevant.len() as f64
}
