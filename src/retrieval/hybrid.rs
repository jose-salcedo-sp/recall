//! `Retriever` over Postgres: full-text and pgvector candidates merged with reciprocal rank fusion.
//!
//! Full text has two sides, both generated columns on `chunks`: `fts_prose` (English stemming, for questions written
//! in prose) and `fts_tokens` (exact lowercased `[a-z0-9_]` tokens, for SAP identifiers such as `S_TCODE`, which the
//! English configuration splits and stems). Prose words are OR-ed; identifier-shaped query tokens must match exactly
//! and dominate the full-text ranking. Within the same identifier-match count, chunks rank by BM25 over `fts_prose`,
//! with document frequencies and the average chunk length from the corpus statistics that `drew ingest` refreshes
//! (`fts_prose_terms`, `fts_prose_corpus`).
//!
//! Before searching, the question gains the words of the glossary rows it matches ([`super::glossary`]); both the
//! full-text query and the embedded query use the expanded text.

use std::{collections::HashMap, fmt, sync::Arc};

use async_trait::async_trait;
use eros::{Context, error};
use pgvector::Vector;
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Postgres, Transaction};

use crate::{
    domain::{Chunk, ChunkId, ChunkRef, ChunkVersion, DocumentId, Labels, Sensitivity, SourceKind},
    llm::{EmbedPurpose, Embedder, ModelRef},
};

use super::{Relevance, Retrieval, Retrieved, Retriever, glossary::Glossary};

/// Retrieval knobs; `drew eval` tunes them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HybridConfig {
    /// Full-text candidates fed to fusion; 0 disables the full-text side.
    pub full_text_candidates: usize,
    /// Vector candidates fed to fusion; 0 disables the vector side. HNSW returns at most 1000.
    pub vector_candidates: usize,
    /// A candidate at 1-based rank `r` in one list contributes `1 / (rrf_k + r)` to its fused score.
    pub rrf_k: f64,
    /// Fused hits kept before expansion.
    pub fused_limit: usize,
    /// Expansion stops adding neighbouring sections to a page once it has this many chunks in the result.
    pub max_chunks_per_document: usize,
    /// Full-text ranking constants.
    pub bm25: Bm25,
    /// Question expansion through the corpus's glossaries.
    pub expansion: ExpansionConfig,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            full_text_candidates: 50,
            vector_candidates: 50,
            rrf_k: 60.0,
            fused_limit: 50,
            max_chunks_per_document: 3,
            bm25: Bm25::default(),
            expansion: ExpansionConfig::default(),
        }
    }
}

/// [`HybridConfig::expansion`]: which documents are glossaries, and how many terms a question may gain from them. The
/// question and its added terms are both the full-text query and the embedded query.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExpansionConfig {
    /// Documents whose pipe tables are read as a [`Glossary`], in this order; none disables expansion.
    pub documents: Vec<DocumentId>,
    /// Most terms one question gains; 0 disables expansion.
    pub max_terms: usize,
}

impl Default for ExpansionConfig {
    fn default() -> Self {
        Self {
            documents: Vec::new(),
            max_terms: 16,
        }
    }
}

/// BM25 constants: `k1` (finite, `>= 0`) is how fast repeats of a term stop adding score, and `b` (in `0..=1`) is
/// how much a chunk longer than average is penalised. Defaults are Lucene's, 1.2 and 0.75.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "Bm25Fields", into = "Bm25Fields")]
pub struct Bm25 {
    k1: f64,
    b: f64,
}

/// Constants rejected by [`Bm25::new`].
#[derive(Debug, PartialEq, thiserror::Error)]
#[error("bm25 needs a finite k1 >= 0 and b in 0..=1, got k1 = {k1}, b = {b}")]
pub struct InvalidBm25 {
    /// The rejected `k1`.
    pub k1: f64,
    /// The rejected `b`.
    pub b: f64,
}

impl Bm25 {
    /// # Errors
    ///
    /// [`InvalidBm25`] when `k1` is negative or not finite, or `b` is outside `0..=1`.
    pub fn new(k1: f64, b: f64) -> Result<Self, InvalidBm25> {
        if k1.is_finite() && k1 >= 0.0 && (0.0..=1.0).contains(&b) {
            Ok(Self { k1, b })
        } else {
            Err(InvalidBm25 { k1, b })
        }
    }

    /// Term-frequency saturation, finite and `>= 0`.
    pub fn k1(self) -> f64 {
        self.k1
    }

    /// Length normalisation, in `0..=1`.
    pub fn b(self) -> f64 {
        self.b
    }
}

impl Default for Bm25 {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Bm25Fields {
    k1: f64,
    b: f64,
}

impl Default for Bm25Fields {
    fn default() -> Self {
        Bm25::default().into()
    }
}

impl From<Bm25> for Bm25Fields {
    fn from(Bm25 { k1, b }: Bm25) -> Self {
        Self { k1, b }
    }
}

impl TryFrom<Bm25Fields> for Bm25 {
    type Error = InvalidBm25;

    fn try_from(Bm25Fields { k1, b }: Bm25Fields) -> Result<Self, InvalidBm25> {
        Self::new(k1, b)
    }
}

/// Current chunks were embedded with a model other than the retriever's embedder. Vectors from different models are
/// not comparable, so the corpus must be re-embedded; retrieval never mixes models.
#[derive(Debug, thiserror::Error)]
#[error(
    "current chunks are embedded with {}, but queries are embedded with {}; re-embed the corpus",
    Models(.found),
    Models(std::slice::from_ref(.expected))
)]
pub struct EmbeddingModelMismatch {
    /// The query embedder's model.
    pub expected: ModelRef,
    /// The first and last distinct models among current chunks (one entry when the corpus is uniform).
    pub found: Vec<ModelRef>,
}

struct Models<'a>(&'a [ModelRef]);

impl fmt::Display for Models<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, model) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{}@{}", model.id, model.version)?;
        }
        Ok(())
    }
}

/// The production [`Retriever`]. Each retrieval first checks that every current chunk uses `embedder`'s model.
pub struct HybridRetriever {
    /// Pool with migrations applied.
    pub pool: PgPool,
    /// Embeds the query; must be the model the corpus was ingested with.
    pub embedder: Arc<dyn Embedder>,
    /// Candidate counts and fusion constants.
    pub config: HybridConfig,
}

impl HybridRetriever {
    /// See the field docs.
    pub fn new(pool: PgPool, embedder: Arc<dyn Embedder>, config: HybridConfig) -> Self {
        Self {
            pool,
            embedder,
            config,
        }
    }

    /// Fails with `EmbeddingModelMismatch` unless every current chunk uses the embedder's model.
    /// Returns false when there are no current chunks.
    #[eros::context("checking the embedding model of current chunks")]
    async fn check_embedding_model(&self) -> eros::Result<bool> {
        let rows: Vec<(String, String)> = sqlx::query_as(
            "(SELECT embedding_model, embedding_model_version FROM chunks
              WHERE superseded_at IS NULL ORDER BY 1, 2 LIMIT 1)
             UNION ALL
             (SELECT embedding_model, embedding_model_version FROM chunks
              WHERE superseded_at IS NULL ORDER BY 1 DESC, 2 DESC LIMIT 1)",
        )
        .fetch_all(&self.pool)
        .await?;
        let mut found: Vec<ModelRef> = rows
            .into_iter()
            .map(|(id, version)| ModelRef::new(id, version))
            .collect();
        found.dedup();
        let expected = self.embedder.model();
        if found.iter().any(|model| *model != expected) {
            return Err(EmbeddingModelMismatch { expected, found }.into());
        }
        Ok(!found.is_empty())
    }

    /// The current chunks of the configured glossary documents, parsed; empty when expansion is off.
    #[eros::context("reading the glossary")]
    async fn glossary(&self) -> eros::Result<Glossary> {
        let config = &self.config.expansion;
        if config.documents.is_empty() || config.max_terms == 0 {
            return Ok(Glossary::default());
        }
        let documents: Vec<&str> = config.documents.iter().map(|id| id.0.as_str()).collect();
        let rows: Vec<(String, String, String)> = sqlx::query_as(
            "SELECT id, version, text FROM chunks
             WHERE superseded_at IS NULL AND document_id = ANY($1)
             ORDER BY array_position($1, document_id), ordinal",
        )
        .bind(&documents)
        .fetch_all(&self.pool)
        .await?;
        Ok(Glossary::parse(rows.iter().map(|(id, version, text)| {
            let source = ChunkRef {
                id: ChunkId(id.clone()),
                version: ChunkVersion(version.clone()),
            };
            (source, text.as_str())
        })))
    }

    /// Full-text candidates for the prose words of `prose` (the question with its glossary expansion) and the
    /// identifiers of `question`. Identifiers come from the question alone, since an identifier match ranks first.
    #[eros::context("full-text candidates")]
    async fn full_text_candidates(
        &self,
        tx: &mut Transaction<'_, Postgres>,
        question: &str,
        prose: &str,
    ) -> eros::Result<Vec<ChunkId>> {
        if self.config.full_text_candidates == 0 {
            return Ok(Vec::new());
        }
        let identifiers = identifier_terms(question);
        // Identifier terms are [a-z0-9_] only, so quoting them cannot break tsquery syntax.
        let identifier_query = (!identifiers.is_empty()).then(|| {
            identifiers
                .iter()
                .map(|term| format!("'{term}'"))
                .collect::<Vec<_>>()
                .join(" | ")
        });
        // The question is tokenized like the chunks (`to_tsvector`), so compound lexemes such as `built-in` or a file
        // path match. Its lexemes are OR-ed as quoted tsquery operands: quotes double and backslashes escape.
        // BM25 uses Lucene's idf. A lexeme new since the last stats refresh counts as in no chunk (the highest idf).
        // With no current chunks at the last refresh, every chunk counts as average length.
        let ids: Vec<String> = sqlx::query_scalar(
            "WITH terms AS (
                 SELECT tsvector_to_array(to_tsvector('english', $1)) AS terms
             ),
             q AS (
                 SELECT terms.terms,
                        (SELECT string_agg(
                                    '''' || replace(replace(t, '\\', '\\\\'), '''', '''''') || '''', ' | ')
                         FROM unnest(terms.terms) t)::tsquery AS prose,
                        $2::tsquery AS tokens
                 FROM terms
             ),
             corpus AS (
                 SELECT coalesce(max(chunks), 0)::float8 AS chunks, max(avg_length) AS avg_length
                 FROM fts_prose_corpus
             )
             SELECT c.id
             FROM chunks c CROSS JOIN q CROSS JOIN corpus
             CROSS JOIN LATERAL (
                 SELECT coalesce(sum(
                            ln(1 + (corpus.chunks - coalesce(t.chunks, 0) + 0.5) / (coalesce(t.chunks, 0) + 0.5))
                            * u.tf * ($5 + 1)
                            / (u.tf + $5 * (1 - $6 + $6 * u.length
                                                 / coalesce(nullif(corpus.avg_length, 0), u.length)))
                        ), 0) AS score
                 FROM (
                     SELECT lexeme, coalesce(array_length(positions, 1), 1) AS tf,
                            sum(coalesce(array_length(positions, 1), 1)) OVER () AS length
                     FROM unnest(c.fts_prose)
                 ) u
                 LEFT JOIN fts_prose_terms t ON t.lexeme = u.lexeme
                 WHERE u.lexeme = ANY(q.terms)
             ) bm25
             WHERE c.superseded_at IS NULL
               AND (c.fts_prose @@ q.prose OR c.fts_tokens @@ q.tokens)
             ORDER BY length(c.fts_tokens) - length(ts_delete(c.fts_tokens, $3::text[])) DESC,
                      bm25.score DESC,
                      c.id
             LIMIT $4",
        )
        .bind(prose)
        .bind(identifier_query)
        .bind(&identifiers)
        .bind(i64::try_from(self.config.full_text_candidates)?)
        .bind(self.config.bm25.k1())
        .bind(self.config.bm25.b())
        .fetch_all(&mut **tx)
        .await?;
        Ok(ids.into_iter().map(ChunkId).collect())
    }

    #[eros::context("vector candidates")]
    async fn vector_candidates(
        &self,
        tx: &mut Transaction<'_, Postgres>,
        embedding: Vec<f32>,
    ) -> eros::Result<Vec<ChunkId>> {
        // HNSW returns at most `ef_search` rows, and pgvector's default (40) is below the default candidate count.
        let ef_search = (2 * self.config.vector_candidates).clamp(40, 1000);
        sqlx::query("SELECT set_config('hnsw.ef_search', $1, true)")
            .bind(ef_search.to_string())
            .execute(&mut **tx)
            .await?;
        let ids: Vec<String> = sqlx::query_scalar(
            "SELECT id FROM chunks WHERE superseded_at IS NULL ORDER BY embedding <=> $1 LIMIT $2",
        )
        .bind(Vector::from(embedding))
        .bind(i64::try_from(self.config.vector_candidates)?)
        .fetch_all(&mut **tx)
        .await?;
        Ok(ids.into_iter().map(ChunkId).collect())
    }

    /// Current rows for `hits`, plus the adjacent current chunks (by ordinal) of each hit's document.
    #[eros::context("fetching retrieved chunks")]
    async fn fetch_with_neighbours(
        &self,
        tx: &mut Transaction<'_, Postgres>,
        hits: &[String],
    ) -> eros::Result<Vec<ChunkRow>> {
        Ok(sqlx::query_as(
            "WITH pages AS (
                 SELECT c.document_id, c.ordinal
                 FROM chunks c
                 WHERE c.superseded_at IS NULL AND c.id = ANY($1)
             ),
             neighbours AS (
                 SELECT n.id FROM pages h CROSS JOIN LATERAL (
                     (SELECT id FROM chunks
                      WHERE superseded_at IS NULL AND document_id = h.document_id AND ordinal < h.ordinal
                      ORDER BY ordinal DESC LIMIT 1)
                     UNION ALL
                     (SELECT id FROM chunks
                      WHERE superseded_at IS NULL AND document_id = h.document_id AND ordinal > h.ordinal
                      ORDER BY ordinal LIMIT 1)
                 ) n
             )
             SELECT c.id, c.version, c.document_id, d.source, d.title, c.heading, c.url, c.text,
                    c.authoritative, c.acl, c.customer, c.sensitivity, c.ordinal
             FROM chunks c JOIN documents d ON d.id = c.document_id
             WHERE c.superseded_at IS NULL AND (c.id = ANY($1) OR c.id IN (SELECT id FROM neighbours))",
        )
        .bind(hits)
        .fetch_all(&mut **tx)
        .await?)
    }
}

#[async_trait]
impl Retriever for HybridRetriever {
    async fn retrieve(&self, query: &str) -> eros::Result<Vec<Retrieved>> {
        Ok(self.retrieve_expanded(query).await?.hits)
    }

    async fn retrieve_expanded(&self, query: &str) -> eros::Result<Retrieval> {
        if query.trim().is_empty() || !self.check_embedding_model().await? {
            return Ok(Retrieval::default());
        }
        let expansion = self
            .glossary()
            .await?
            .expand(query, self.config.expansion.max_terms);
        let expanded = expansion.apply(query);
        let embedding = if self.config.vector_candidates > 0 {
            let mut vectors = self
                .embedder
                .embed(std::slice::from_ref(&expanded), EmbedPurpose::Query)
                .await
                .context("embedding the query")?;
            Some(vectors.pop().context("embedder returned no vector")?)
        } else {
            None
        };

        let mut tx = self.pool.begin().await?;
        sqlx::query("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            .execute(&mut *tx)
            .await?;
        let full_text = self.full_text_candidates(&mut tx, query, &expanded).await?;
        let vector = match embedding {
            Some(embedding) => self.vector_candidates(&mut tx, embedding).await?,
            None => Vec::new(),
        };
        let mut fused = reciprocal_rank_fusion(&[&full_text, &vector], self.config.rrf_k);
        fused.truncate(self.config.fused_limit);
        let ids: Vec<String> = fused.iter().map(|(id, _)| id.0.clone()).collect();
        let rows = self.fetch_with_neighbours(&mut tx, &ids).await?;
        tx.commit().await?;

        Ok(Retrieval {
            hits: expand(&fused, rows, self.config.max_chunks_per_document)?,
            expansion,
        })
    }
}

/// Fuses ranked lists: a chunk at 1-based rank `r` in a list gains `1 / (k + r)`. Best first; ties go to the better
/// single-list rank, then to the smaller chunk id.
pub fn reciprocal_rank_fusion(lists: &[&[ChunkId]], k: f64) -> Vec<(ChunkId, f64)> {
    let mut fused: HashMap<&ChunkId, (f64, usize)> = HashMap::new();
    for list in lists {
        for (index, id) in list.iter().enumerate() {
            let rank = index + 1;
            let entry = fused.entry(id).or_insert((0.0, rank));
            entry.0 += 1.0 / (k + rank as f64);
            entry.1 = entry.1.min(rank);
        }
    }
    let mut fused: Vec<_> = fused.into_iter().collect();
    fused.sort_by(|(a_id, (a_score, a_rank)), (b_id, (b_score, b_rank))| {
        b_score
            .total_cmp(a_score)
            .then(a_rank.cmp(b_rank))
            .then(a_id.cmp(b_id))
    });
    fused
        .into_iter()
        .map(|(id, (score, _))| (id.clone(), score))
        .collect()
}

/// Expands fused hits (best first) to their parent units and dedupes by chunk id.
///
/// Every hit is kept. A hit also pulls in the previous and next current chunks of its document (ordinal ±1 when
/// ordinals are dense): the neighbouring sections of a markdown file. Each is scored one step below the hit, while
/// the document has fewer than `max_per_document` chunks in the result; the cap never drops a hit. A file with a
/// single section has no neighbours.
/// A chunk reached twice keeps its best score, and the result is sorted best first.
fn expand(
    hits: &[(ChunkId, f64)],
    rows: Vec<ChunkRow>,
    max_per_document: usize,
) -> eros::Result<Vec<Retrieved>> {
    let mut chunks = HashMap::new();
    let mut pages: HashMap<DocumentId, Vec<(i32, ChunkId)>> = HashMap::new();
    for row in rows {
        let ordinal = row.ordinal;
        let chunk = Chunk::try_from(row)?;
        pages
            .entry(chunk.document_id.clone())
            .or_default()
            .push((ordinal, chunk.id.clone()));
        chunks.insert(chunk.id.clone(), chunk);
    }
    pages.values_mut().for_each(|page| page.sort());

    let mut result = Expansion::default();
    for (id, fused) in hits {
        let hit = chunks
            .get(id)
            .ok_or_else(|| error!("hit {} has no current row", id.0))?;
        let score = *fused as f32;
        result.add(hit, score, None);
        let Some(page) = pages.get(&hit.document_id) else {
            continue;
        };
        let at = page
            .iter()
            .position(|(_, chunk)| chunk == id)
            .ok_or_else(|| error!("hit {} missing from its page", id.0))?;
        let neighbours = [at.checked_sub(1), Some(at + 1)]
            .into_iter()
            .flatten()
            .filter_map(|i| page.get(i));
        for (_, neighbour) in neighbours {
            if result.contains(neighbour) {
                continue;
            }
            if result.count(&hit.document_id) >= max_per_document {
                break;
            }
            result.add(&chunks[neighbour], score.next_down(), Some(id));
        }
    }
    let mut out = result.out;
    out.sort_by(|a, b| b.score.0.total_cmp(&a.score.0));
    Ok(out)
}

#[derive(Default)]
struct Expansion {
    out: Vec<Retrieved>,
    position: HashMap<ChunkId, usize>,
    per_document: HashMap<DocumentId, usize>,
}

impl Expansion {
    /// A chunk that is a match anywhere stays a match, even if it was first added as another match's context.
    fn add(&mut self, chunk: &Chunk, score: f32, context_for: Option<&ChunkId>) {
        match self.position.get(&chunk.id) {
            Some(&i) => {
                let existing = &mut self.out[i];
                existing.score = Relevance(existing.score.0.max(score));
                if context_for.is_none() {
                    existing.context_for = None;
                }
            }
            None => {
                self.position.insert(chunk.id.clone(), self.out.len());
                *self
                    .per_document
                    .entry(chunk.document_id.clone())
                    .or_default() += 1;
                self.out.push(Retrieved {
                    chunk: chunk.clone(),
                    score: Relevance(score),
                    context_for: context_for.cloned(),
                });
            }
        }
    }

    fn contains(&self, id: &ChunkId) -> bool {
        self.position.contains_key(id)
    }

    fn count(&self, document: &DocumentId) -> usize {
        self.per_document.get(document).copied().unwrap_or(0)
    }
}

/// Identifier-shaped query tokens, matched exactly against `chunks.fts_tokens`: tokens over `[A-Za-z0-9_]` (edge
/// underscores trimmed, as in the migration) that contain a letter and either an underscore or a digit, such as
/// `S_TCODE`, `AGR_USERS`, or `SU24`. Lowercased, sorted, and deduplicated.
pub fn identifier_terms(query: &str) -> Vec<String> {
    let mut terms: Vec<String> = query
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .map(|token| token.trim_matches('_'))
        .filter(|token| {
            token.len() < 64
                && token.chars().any(|c| c.is_ascii_alphabetic())
                && token.chars().any(|c| c == '_' || c.is_ascii_digit())
        })
        .map(str::to_ascii_lowercase)
        .collect();
    terms.sort();
    terms.dedup();
    terms
}

#[derive(sqlx::FromRow)]
struct ChunkRow {
    id: String,
    version: String,
    document_id: String,
    source: String,
    title: String,
    heading: Option<String>,
    url: String,
    text: String,
    authoritative: bool,
    acl: Vec<String>,
    customer: Option<String>,
    sensitivity: String,
    ordinal: i32,
}

impl TryFrom<ChunkRow> for Chunk {
    type Error = eros::ErrorUnion;

    fn try_from(row: ChunkRow) -> eros::Result<Self> {
        let source = SourceKind::try_from(row.source.as_str())
            .with_context(|| format!("chunk {}", row.id))?;
        let sensitivity = Sensitivity::try_from(row.sensitivity.as_str())
            .with_context(|| format!("chunk {}", row.id))?;
        Ok(Chunk {
            id: ChunkId(row.id),
            version: ChunkVersion(row.version),
            document_id: DocumentId(row.document_id),
            source,
            title: row.title,
            heading: row.heading,
            url: row.url,
            text: row.text,
            authoritative: row.authoritative,
            labels: Labels {
                acl: row.acl,
                customer: row.customer,
                sensitivity,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(names: &[&str]) -> Vec<ChunkId> {
        names.iter().map(|name| ChunkId(name.to_string())).collect()
    }

    #[test]
    fn fusion_rewards_agreement_between_lists() {
        let full_text = ids(&["x", "z"]);
        let vector = ids(&["y", "z", "f1", "f2", "x"]);
        let fused = reciprocal_rank_fusion(&[&full_text, &vector], 60.0);
        let order: Vec<_> = fused.iter().map(|(id, _)| id.0.as_str()).collect();
        assert_eq!(order, ["z", "x", "y", "f1", "f2"]);
        assert!((fused[0].1 - 2.0 / 62.0).abs() < 1e-12);
        assert!((fused[1].1 - (1.0 / 61.0 + 1.0 / 65.0)).abs() < 1e-12);
        assert!((fused[2].1 - 1.0 / 61.0).abs() < 1e-12);
    }

    #[test]
    fn fusion_ties_break_by_best_rank_then_id() {
        let full_text = ids(&["b", "c"]);
        let vector = ids(&["a", "d"]);
        let fused = reciprocal_rank_fusion(&[&full_text, &vector], 60.0);
        let order: Vec<_> = fused.iter().map(|(id, _)| id.0.as_str()).collect();
        assert_eq!(order, ["a", "b", "c", "d"]);
    }

    #[test]
    fn identifier_terms_keep_sap_names_whole() {
        assert_eq!(
            identifier_terms("Does S_TCODE gate _AGR_USERS_ or SU24, not su24? PFCG 2024"),
            ["agr_users", "s_tcode", "su24"]
        );
        assert!(identifier_terms("which table maps users to roles").is_empty());
    }
}
