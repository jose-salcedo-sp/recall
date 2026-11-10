//! Retrieval does not judge relevance: a non-blank query over a non-empty corpus returns its nearest chunks, and
//! Rerank decides which of them answer the question.

use std::collections::HashSet;

use drew::{
    domain::{Chunk, ChunkId, ChunkVersion, DocumentId, Labels, Sensitivity, SourceKind},
    retrieval::Retriever,
};

const QUESTION: &str = "Which table maps users to roles?";

/// One chunk in a document of its own, so no retriever adds neighbouring sections as context.
fn chunk(name: &str, text: &str) -> Chunk {
    let document_id = DocumentId::new(SourceKind::Markdown, &format!("{name}.md"));
    Chunk {
        id: ChunkId::new(&document_id, name),
        version: ChunkVersion("v1".into()),
        document_id,
        source: SourceKind::Markdown,
        title: format!("{name} notes"),
        heading: Some(name.into()),
        url: format!("https://wiki.example.test/{name}"),
        text: text.into(),
        authoritative: true,
        labels: Labels {
            acl: Vec::new(),
            customer: None,
            sensitivity: Sensitivity::Internal,
        },
    }
}

fn agr_users() -> Chunk {
    chunk(
        "agr-users",
        "AGR_USERS maps users to roles, with validity dates per assignment.",
    )
}

fn corpus() -> Vec<Chunk> {
    vec![
        agr_users(),
        chunk(
            "usr02",
            "USR02 stores user logon data; it does not map roles.",
        ),
        chunk("lunch", "Lunch is at noon on Fridays."),
    ]
}

fn ids(chunks: &[Chunk]) -> HashSet<ChunkId> {
    chunks.iter().map(|chunk| chunk.id.clone()).collect()
}

pub async fn an_empty_corpus_returns_nothing(
    make: impl AsyncFnOnce(Vec<Chunk>) -> Box<dyn Retriever>,
) -> eros::Result<()> {
    let retriever = make(Vec::new()).await;

    assert!(retriever.retrieve(QUESTION).await?.is_empty());
    Ok(())
}

pub async fn a_blank_query_returns_nothing(
    make: impl AsyncFnOnce(Vec<Chunk>) -> Box<dyn Retriever>,
) -> eros::Result<()> {
    let retriever = make(corpus()).await;

    assert!(retriever.retrieve("").await?.is_empty());
    assert!(retriever.retrieve(" \n\t").await?.is_empty());
    Ok(())
}

pub async fn an_unrelated_query_still_returns_the_nearest_chunks(
    make: impl AsyncFnOnce(Vec<Chunk>) -> Box<dyn Retriever>,
) -> eros::Result<()> {
    let retriever = make(corpus()).await;

    for query in ["Cafeteria opening hours?", "???"] {
        let found: Vec<_> = retriever
            .retrieve(query)
            .await?
            .into_iter()
            .map(|hit| hit.chunk)
            .collect();
        assert_eq!(ids(&found), ids(&corpus()), "{query:?}");
    }
    Ok(())
}

pub async fn the_best_match_ranks_first(
    make: impl AsyncFnOnce(Vec<Chunk>) -> Box<dyn Retriever>,
) -> eros::Result<()> {
    let retriever = make(corpus()).await;

    let hits = retriever.retrieve(QUESTION).await?;

    assert_eq!(hits[0].chunk.id, agr_users().id);
    Ok(())
}

pub async fn hits_are_best_first_without_duplicates(
    make: impl AsyncFnOnce(Vec<Chunk>) -> Box<dyn Retriever>,
) -> eros::Result<()> {
    let retriever = make(corpus()).await;

    let hits = retriever.retrieve(QUESTION).await?;

    assert!(
        hits.windows(2)
            .all(|pair| pair[0].score.0 >= pair[1].score.0),
        "{hits:#?}"
    );
    let unique: HashSet<_> = hits.iter().map(|hit| &hit.chunk.id).collect();
    assert_eq!(unique.len(), hits.len());
    Ok(())
}

pub async fn equal_matches_rank_by_chunk_id(
    make: impl AsyncFnOnce(Vec<Chunk>) -> Box<dyn Retriever>,
) -> eros::Result<()> {
    let text = "PFCG maintains roles.";
    let (later, earlier) = (chunk("pfcg-b", text), chunk("pfcg-a", text));
    let retriever = make(vec![later.clone(), earlier.clone()]).await;

    let hits = retriever.retrieve("PFCG maintains roles").await?;

    let order: Vec<_> = hits.iter().map(|hit| &hit.chunk.id).collect();
    assert_eq!(order, [&earlier.id, &later.id]);
    Ok(())
}

pub async fn chunks_come_back_as_stored(
    make: impl AsyncFnOnce(Vec<Chunk>) -> Box<dyn Retriever>,
) -> eros::Result<()> {
    let retriever = make(corpus()).await;

    let hits = retriever.retrieve(QUESTION).await?;

    for hit in &hits {
        let stored = corpus()
            .into_iter()
            .find(|chunk| chunk.id == hit.chunk.id)
            .expect("every hit is from the corpus");
        assert_eq!(hit.chunk, stored);
        assert_eq!(hit.context_for, None);
    }
    Ok(())
}

macro_rules! retriever_contract {
    (#[$attr:meta] $args:tt => $make:expr) => {
        retriever_contract!(@cases #[$attr] $args => $make;
            an_empty_corpus_returns_nothing,
            a_blank_query_returns_nothing,
            an_unrelated_query_still_returns_the_nearest_chunks,
            the_best_match_ranks_first,
            hits_are_best_first_without_duplicates,
            equal_matches_rank_by_chunk_id,
            chunks_come_back_as_stored,
        );
    };
    (@cases #[$attr:meta] $args:tt => $make:expr; $($case:ident,)*) => {
        $(
            #[$attr]
            async fn $case $args -> eros::Result<()> {
                crate::retriever::$case($make).await
            }
        )*
    };
}

mod keyword {
    use drew::{
        domain::Chunk,
        retrieval::{Retriever, fakes::KeywordRetriever},
    };

    async fn keyword(chunks: Vec<Chunk>) -> Box<dyn Retriever> {
        Box::new(KeywordRetriever { chunks })
    }

    retriever_contract!(#[tokio::test] () => keyword);
}

mod hybrid {
    use std::sync::Arc;

    use drew::{
        domain::Chunk,
        llm::{EmbedPurpose, Embedder, fakes::HashEmbedder},
        retrieval::{
            Retriever,
            hybrid::{HybridConfig, HybridRetriever},
        },
    };
    use pgvector::Vector;
    use sqlx::PgPool;

    async fn seeded(pool: PgPool, chunks: Vec<Chunk>) -> Box<dyn Retriever> {
        for chunk in &chunks {
            insert(&pool, chunk).await;
        }
        Box::new(HybridRetriever::new(
            pool,
            Arc::new(HashEmbedder),
            HybridConfig::default(),
        ))
    }

    async fn insert(pool: &PgPool, chunk: &Chunk) {
        sqlx::query(
            "INSERT INTO documents (id, source, external_id, title, url, source_updated_at)
             VALUES ($1, $2, $3, $4, $5, now())",
        )
        .bind(&chunk.document_id.0)
        .bind(chunk.source.as_str())
        .bind(&chunk.document_id.0)
        .bind(&chunk.title)
        .bind(&chunk.url)
        .execute(pool)
        .await
        .unwrap();
        let embedding = HashEmbedder
            .embed(std::slice::from_ref(&chunk.text), EmbedPurpose::Document)
            .await
            .unwrap()
            .remove(0);
        let model = HashEmbedder.model();
        sqlx::query(
            "INSERT INTO chunks (id, version, document_id, ordinal, heading, url, text, authoritative, acl,
                                 customer, sensitivity, embedding, embedding_model, embedding_model_version)
             VALUES ($1, $2, $3, 0, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)",
        )
        .bind(&chunk.id.0)
        .bind(&chunk.version.0)
        .bind(&chunk.document_id.0)
        .bind(&chunk.heading)
        .bind(&chunk.url)
        .bind(&chunk.text)
        .bind(chunk.authoritative)
        .bind(&chunk.labels.acl)
        .bind(&chunk.labels.customer)
        .bind(chunk.labels.sensitivity.as_str())
        .bind(Vector::from(embedding))
        .bind(&model.id)
        .bind(&model.version)
        .execute(pool)
        .await
        .unwrap();
    }

    retriever_contract!(
        #[sqlx::test(migrator = "drew::store::MIGRATOR")] (pool: PgPool) => |chunks| seeded(pool, chunks)
    );
}
