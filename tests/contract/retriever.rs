//! Retrieval does not judge relevance: a non-blank query over a non-empty corpus returns its nearest chunks, and
//! Rerank decides which of them answer the question. It does judge access: only chunks the scope admits come back,
//! as matches or as context.

use std::collections::HashSet;

use recall::{
    access::{Acl, AclEntry, ChunkGrant, ChunkScope, Role, fakes::unrestricted},
    domain::{Chunk, ChunkId, ChunkVersion, DocumentId, Labels, Sensitivity, UserId},
    ingest::markdown,
    retrieval::Retriever,
};

fn everything() -> ChunkScope {
    unrestricted().chunks().clone()
}

const QUESTION: &str = "Which table maps users to roles?";

/// One chunk in a document of its own, so no retriever adds neighbouring sections as context.
fn chunk(name: &str, text: &str) -> Chunk {
    let document_id = DocumentId::new(&markdown::NAME, &format!("{name}.md"));
    Chunk {
        id: ChunkId::new(&document_id, name),
        version: ChunkVersion("v1".into()),
        document_id,
        source: markdown::NAME.clone(),
        title: format!("{name} notes"),
        heading: Some(name.into()),
        url: format!("https://wiki.example.test/{name}"),
        text: text.into(),
        authoritative: true,
        labels: Labels {
            acl: Acl::anyone(),
            sensitivity: Sensitivity::Internal,
        },
    }
}

fn role_users() -> Chunk {
    chunk(
        "role-users",
        "ROLE_USERS maps users to roles, with validity dates per assignment.",
    )
}

fn corpus() -> Vec<Chunk> {
    vec![
        role_users(),
        chunk(
            "login02",
            "LOGIN02 stores user logon data; it does not map roles.",
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

    assert!(
        retriever
            .retrieve(QUESTION, &everything())
            .await?
            .is_empty()
    );
    Ok(())
}

pub async fn a_blank_query_returns_nothing(
    make: impl AsyncFnOnce(Vec<Chunk>) -> Box<dyn Retriever>,
) -> eros::Result<()> {
    let retriever = make(corpus()).await;

    assert!(retriever.retrieve("", &everything()).await?.is_empty());
    assert!(retriever.retrieve(" \n\t", &everything()).await?.is_empty());
    Ok(())
}

pub async fn an_unrelated_query_still_returns_the_nearest_chunks(
    make: impl AsyncFnOnce(Vec<Chunk>) -> Box<dyn Retriever>,
) -> eros::Result<()> {
    let retriever = make(corpus()).await;

    for query in ["Cafeteria opening hours?", "???"] {
        let found: Vec<_> = retriever
            .retrieve(query, &everything())
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

    let hits = retriever.retrieve(QUESTION, &everything()).await?;

    assert_eq!(hits[0].chunk.id, role_users().id);
    Ok(())
}

pub async fn hits_are_best_first_without_duplicates(
    make: impl AsyncFnOnce(Vec<Chunk>) -> Box<dyn Retriever>,
) -> eros::Result<()> {
    let retriever = make(corpus()).await;

    let hits = retriever.retrieve(QUESTION, &everything()).await?;

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
    let text = "ROLEADM maintains roles.";
    let (later, earlier) = (chunk("pfcg-b", text), chunk("pfcg-a", text));
    let retriever = make(vec![later.clone(), earlier.clone()]).await;

    let hits = retriever
        .retrieve("ROLEADM maintains roles", &everything())
        .await?;

    let order: Vec<_> = hits.iter().map(|hit| &hit.chunk.id).collect();
    assert_eq!(order, [&earlier.id, &later.id]);
    Ok(())
}

pub async fn chunks_come_back_as_stored(
    make: impl AsyncFnOnce(Vec<Chunk>) -> Box<dyn Retriever>,
) -> eros::Result<()> {
    let retriever = make(corpus()).await;

    let hits = retriever.retrieve(QUESTION, &everything()).await?;

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

fn labelled(document: &str, locator: &str, text: &str, labels: Labels) -> Chunk {
    let document_id = DocumentId::new(&markdown::NAME, &format!("{document}.md"));
    Chunk {
        id: ChunkId::new(&document_id, locator),
        labels,
        document_id,
        title: format!("{document} notes"),
        ..chunk(locator, text)
    }
}

fn labels(sensitivity: Sensitivity, acl: &[&str]) -> Labels {
    Labels {
        acl: Acl::only(acl.iter().map(|entry| entry.parse::<AclEntry>().unwrap())),
        sensitivity,
    }
}

/// Every chunk is about the question, so only the scope decides which come back. `roles` has an internal section
/// followed by a confidential one, which a retriever that adds neighbouring sections would add as context.
fn labelled_corpus() -> Vec<Chunk> {
    let about = "ROLE_USERS maps users to roles.";
    vec![
        labelled("roles", "public", about, labels(Sensitivity::Internal, &[])),
        labelled(
            "roles",
            "secret",
            "ROLE_USERS overrides: ROLE_ROOT maps users to every role.",
            labels(Sensitivity::Confidential, &[]),
        ),
        labelled(
            "ana",
            "plant",
            about,
            labels(Sensitivity::Internal, &["user:ana"]),
        ),
        labelled(
            "hr",
            "pay",
            about,
            labels(Sensitivity::Public, &["role:hr"]),
        ),
        labelled(
            "eve",
            "notes",
            about,
            labels(Sensitivity::Public, &["user:eve"]),
        ),
        labelled("faq", "open", about, labels(Sensitivity::Public, &[])),
    ]
}

fn scope(user: &str, roles: &[&str], clearance: Sensitivity) -> ChunkScope {
    ChunkScope::new(
        UserId::new(user).unwrap(),
        roles.iter().map(|role| Role::new(*role).unwrap()),
        [ChunkGrant::new(Role::new("reader").unwrap(), clearance)],
    )
}

fn scopes() -> Vec<(&'static str, ChunkScope)> {
    vec![
        ("none", ChunkScope::none()),
        ("internal", scope("eve", &[], Sensitivity::Internal)),
        ("public", scope("eve", &[], Sensitivity::Public)),
        ("ana", scope("ana", &[], Sensitivity::Internal)),
        ("hr", scope("ana", &["hr"], Sensitivity::Public)),
        (
            "a user named hr",
            scope("hr", &["eve"], Sensitivity::Public),
        ),
        ("everything", everything()),
    ]
}

/// Exactly the readable chunks come back, as matches or as context: a chunk outside the scope is not returned, and
/// none takes the place of a readable one.
pub async fn only_chunks_the_scope_admits_come_back(
    make: impl AsyncFnOnce(Vec<Chunk>) -> Box<dyn Retriever>,
) -> eros::Result<()> {
    let corpus = labelled_corpus();
    let retriever = make(corpus.clone()).await;

    for (name, scope) in scopes() {
        let hits = retriever.retrieve(QUESTION, &scope).await?;
        let found: HashSet<_> = hits.iter().map(|hit| hit.chunk.id.clone()).collect();
        let readable: HashSet<_> = corpus
            .iter()
            .filter(|chunk| scope.admits(&chunk.labels))
            .map(|chunk| chunk.id.clone())
            .collect();
        assert_eq!(found, readable, "{name}");
        for hit in &hits {
            assert!(
                scope.admits(&hit.chunk.labels),
                "{name}: {}",
                hit.chunk.id.0
            );
        }
    }
    Ok(())
}

/// A user entry matches only that user and a role entry only a holder of the role: a principal whose id is `hr`
/// does not read a chunk ACL'd to the role `hr`, and its role `eve` does not read a chunk ACL'd to the user `eve`.
pub async fn a_user_id_never_matches_a_role_entry(
    make: impl AsyncFnOnce(Vec<Chunk>) -> Box<dyn Retriever>,
) -> eros::Result<()> {
    let retriever = make(labelled_corpus()).await;
    let clash = scope("hr", &["eve"], Sensitivity::Public);

    let found: HashSet<String> = retriever
        .retrieve(QUESTION, &clash)
        .await?
        .into_iter()
        .map(|hit| hit.chunk.id.0)
        .collect();

    assert!(found.contains("markdown:faq.md#open"), "{found:?}");
    assert!(!found.contains("markdown:hr.md#pay"), "{found:?}");
    assert!(!found.contains("markdown:eve.md#notes"), "{found:?}");
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
            only_chunks_the_scope_admits_come_back,
            a_user_id_never_matches_a_role_entry,
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
    use recall::{
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

    use pgvector::Vector;
    use recall::{
        domain::Chunk,
        llm::{EmbedPurpose, Embedder, fakes::HashEmbedder},
        retrieval::{
            Retriever,
            hybrid::{HybridConfig, HybridRetriever},
        },
    };
    use sqlx::PgPool;

    async fn seeded(pool: PgPool, chunks: Vec<Chunk>) -> Box<dyn Retriever> {
        let mut ordinals = std::collections::HashMap::new();
        for chunk in &chunks {
            let ordinal: &mut i32 = ordinals.entry(chunk.document_id.clone()).or_default();
            insert(&pool, chunk, *ordinal).await;
            *ordinal += 1;
        }
        Box::new(HybridRetriever::new(
            pool,
            Arc::new(HashEmbedder),
            HybridConfig::default(),
        ))
    }

    async fn insert(pool: &PgPool, chunk: &Chunk, ordinal: i32) {
        sqlx::query(
            "INSERT INTO documents (id, source, external_id, title, url, source_updated_at)
             VALUES ($1, $2, $3, $4, $5, now()) ON CONFLICT (id) DO NOTHING",
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
                                 sensitivity, embedding, embedding_model, embedding_model_version)
             VALUES ($1, $2, $3, $13, $4, $5, $6, $7, $8, $9, $10, $11, $12)",
        )
        .bind(&chunk.id.0)
        .bind(&chunk.version.0)
        .bind(&chunk.document_id.0)
        .bind(&chunk.heading)
        .bind(&chunk.url)
        .bind(&chunk.text)
        .bind(chunk.authoritative)
        .bind(chunk.labels.acl.to_strings())
        .bind(chunk.labels.sensitivity.as_str())
        .bind(Vector::from(embedding))
        .bind(&model.id)
        .bind(&model.version)
        .bind(ordinal)
        .execute(pool)
        .await
        .unwrap();
    }

    retriever_contract!(
        #[sqlx::test(migrator = "recall::store::MIGRATOR")] (pool: PgPool) => |chunks| seeded(pool, chunks)
    );
}
