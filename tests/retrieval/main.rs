//! Hybrid retrieval against Postgres.

#![allow(
    clippy::unwrap_used,
    reason = "integration-test helpers; clippy exempts only #[test] functions"
)]

mod seed;

use std::sync::Arc;

use drew::{
    domain::{ChunkId, SourceKind},
    llm::{Embedder, ModelRef, fakes::HashEmbedder},
    retrieval::{
        Relevance, Retrieved, Retriever,
        hybrid::{EmbeddingModelMismatch, ExpansionConfig, HybridConfig, HybridRetriever},
    },
    store::chunks::ChunkStore,
};
use seed::{SeedChunk, document, embed, recall_at_k, similar_to};
use sqlx::{PgPool, postgres::PgPoolOptions};

fn retriever(pool: &PgPool, config: HybridConfig) -> HybridRetriever {
    HybridRetriever::new(pool.clone(), Arc::new(HashEmbedder), config)
}

fn full_text_only() -> HybridConfig {
    HybridConfig {
        vector_candidates: 0,
        ..HybridConfig::default()
    }
}

fn ids(results: &[Retrieved]) -> Vec<&str> {
    results.iter().map(|r| r.chunk.id.0.as_str()).collect()
}

fn position(results: &[Retrieved], id: &ChunkId) -> Option<usize> {
    results.iter().position(|r| r.chunk.id == *id)
}

#[sqlx::test(migrator = "drew::store::MIGRATOR")]
async fn identifier_queries_match_exactly_via_full_text(pool: PgPool) {
    let chunk = async |key: &str, text: &str| {
        let doc = document(&pool, SourceKind::Markdown, key, key).await;
        SeedChunk::new(&doc, "body", 0, text).insert(&pool).await
    };
    let s_tcode = chunk(
        "note-1.md",
        "Start authorization: S_TCODE is checked when a user calls a transaction.",
    )
    .await;
    let tcode_prose = chunk(
        "note-2.md",
        "The s tcode column of the TSTC table lists transactions; see the TCODE field.",
    )
    .await;
    let agr_users = chunk("note-3.md", "AGR_USERS stores role assignments per user.").await;
    let agr_users_t = chunk(
        "note-4.md",
        "AGR_USERS_T holds role texts; agr users read it for descriptions.",
    )
    .await;
    let su24 = chunk(
        "note-5.md",
        "Use SU24 to maintain authorization proposal values.",
    )
    .await;
    let su24n = chunk(
        "note-6.md",
        "SU24N is the newer screen for proposal values.",
    )
    .await;
    let retriever = retriever(&pool, full_text_only());

    // English stems both chunks to `tcode`; only the exact token side tells them apart.
    let results = retriever.retrieve("S_TCODE").await.unwrap();
    assert_eq!(results[0].chunk.id, s_tcode);
    assert!(position(&results, &tcode_prose) > Some(0));

    let results = retriever.retrieve("Who reads AGR_USERS?").await.unwrap();
    assert_eq!(results[0].chunk.id, agr_users);
    assert!(position(&results, &agr_users_t) > Some(0));

    let results = retriever.retrieve("what is su24").await.unwrap();
    assert_eq!(ids(&results), [su24.0.as_str()]);
    assert_eq!(position(&results, &su24n), None);

    let results = retriever
        .retrieve("Which table is _AGR_USERS_?")
        .await
        .unwrap();
    assert_eq!(results[0].chunk.id, agr_users);
}

#[sqlx::test(migrator = "drew::store::MIGRATOR")]
async fn identifiers_in_headings_and_markdown_are_indexed(pool: PgPool) {
    let page = document(&pool, SourceKind::Markdown, "10", "Tables").await;
    let heading = SeedChunk::new(&page, "agr-1251", 0, "Authorization values per role.")
        .heading("AGR_1251")
        .insert(&pool)
        .await;
    let notes = document(&pool, SourceKind::Markdown, "notes.md", "Notes").await;
    let italic = SeedChunk::new(&notes, "body", 0, "check _USOBT_C_ first")
        .insert(&pool)
        .await;
    let retriever = retriever(&pool, full_text_only());

    let results = retriever.retrieve("AGR_1251").await.unwrap();
    assert_eq!(ids(&results), [heading.0.as_str()]);
    let results = retriever.retrieve("usobt_c").await.unwrap();
    assert_eq!(ids(&results), [italic.0.as_str()]);
}

#[sqlx::test(migrator = "drew::store::MIGRATOR")]
async fn a_rare_term_outranks_repeats_of_a_common_one(pool: PgPool) {
    let chunk = async |key: &str, text: &str| {
        let doc = document(&pool, SourceKind::Markdown, key, key).await;
        SeedChunk::new(&doc, "body", 0, text).insert(&pool).await
    };
    let rare = chunk("payroll.md", "Payroll runs at month end.").await;
    let common = chunk(
        "owners.md",
        "Role owners review role changes; every role needs a role owner.",
    )
    .await;
    for i in 0..5 {
        chunk(&format!("note-{i}.md"), &format!("Note {i} on the role.")).await;
    }
    ChunkStore::new(pool.clone())
        .refresh_term_stats()
        .await
        .unwrap();

    let results = retriever(&pool, full_text_only())
        .retrieve("Which role handles payroll?")
        .await
        .unwrap();
    assert_eq!(results[0].chunk.id, rare);
    assert_eq!(results[1].chunk.id, common);
    assert_eq!(results.len(), 7);
}

#[sqlx::test(migrator = "drew::store::MIGRATOR")]
async fn tsquery_syntax_in_a_question_is_plain_text(pool: PgPool) {
    let doc = document(&pool, SourceKind::Markdown, "obrien.md", "Owners").await;
    let id = SeedChunk::new(&doc, "body", 0, "O'Brien owns the built-in roles.")
        .insert(&pool)
        .await;
    let results = retriever(&pool, full_text_only())
        .retrieve(r"O'Brien & !(built-in):* | 'roles\")
        .await
        .unwrap();
    assert_eq!(ids(&results), [id.0.as_str()]);
}

#[sqlx::test(migrator = "drew::store::MIGRATOR")]
async fn prose_query_hits_via_vectors(pool: PgPool) {
    let question = "how do I take away someone's access when they quit";
    let doc = document(&pool, SourceKind::Markdown, "note-7.md", "Leavers").await;
    // Stands in for a semantic embedding: the chunk shares no stem with the question, only meaning.
    let leavers = SeedChunk::new(
        &doc,
        "body",
        0,
        "Revoking authorizations for leavers: lock the user and remove role assignments.",
    )
    .embedding(embed(question).await)
    .insert(&pool)
    .await;
    for (key, text) in [
        ("note-8.md", "Transport roles between clients with SCC1."),
        ("note-9.md", "Payroll cluster tables are read with macros."),
    ] {
        let doc = document(&pool, SourceKind::Markdown, key, key).await;
        SeedChunk::new(&doc, "body", 0, text).insert(&pool).await;
    }

    let full_text = retriever(&pool, full_text_only())
        .retrieve(question)
        .await
        .unwrap();
    assert_eq!(position(&full_text, &leavers), None);

    let hybrid = retriever(&pool, HybridConfig::default())
        .retrieve(question)
        .await
        .unwrap();
    assert_eq!(hybrid[0].chunk.id, leavers);
}

#[sqlx::test(migrator = "drew::store::MIGRATOR")]
async fn fusion_orders_by_reciprocal_rank(pool: PgPool) {
    let query = "S_TCODE authorization check";
    let q = embed(query).await;
    let chunk = async |key: &str, text: &str, cosine: f32| {
        let doc = document(&pool, SourceKind::Markdown, key, key).await;
        SeedChunk::new(&doc, "body", 0, text)
            .embedding(similar_to(&q, cosine))
            .insert(&pool)
            .await
    };
    // Full text: x (identifier), z (prose). Vectors: y, z, f1..f5, x.
    let x = chunk(
        "X-1",
        "S_TCODE authorization check at transaction start",
        0.0,
    )
    .await;
    let z = chunk("Z-1", "Authorization check overview", 0.9).await;
    let y = chunk("Y-1", "Payroll schema reference", 1.0).await;
    let mut fillers = Vec::new();
    for (i, cosine) in [0.8, 0.7, 0.6, 0.5, 0.4].into_iter().enumerate() {
        fillers.push(chunk(&format!("F-{i}"), "Payroll cluster note", cosine).await);
    }

    let results = retriever(&pool, HybridConfig::default())
        .retrieve(query)
        .await
        .unwrap();

    let mut expected = vec![
        (z, 2.0 / 62.0),
        (x, 1.0 / 61.0 + 1.0 / 68.0),
        (y, 1.0 / 61.0),
    ];
    for (i, filler) in fillers.into_iter().enumerate() {
        expected.push((filler, 1.0 / (63.0 + i as f64)));
    }
    assert_eq!(results.len(), expected.len());
    for (result, (id, score)) in results.iter().zip(&expected) {
        assert_eq!(result.chunk.id, *id);
        assert!((f64::from(result.score.0) - score).abs() < 1e-6, "{id:?}");
    }
}

#[sqlx::test(migrator = "drew::store::MIGRATOR")]
async fn superseded_versions_are_never_returned(pool: PgPool) {
    let page = document(&pool, SourceKind::Markdown, "20", "Start authorization").await;
    SeedChunk::new(&page, "intro", 0, "Intro, old wording.")
        .version("old-intro")
        .superseded()
        .insert(&pool)
        .await;
    let intro = SeedChunk::new(&page, "intro", 0, "Intro to start checks.")
        .version("new-intro")
        .insert(&pool)
        .await;
    let removed = SeedChunk::new(&page, "removed", 1, "A section deleted from the page.")
        .superseded()
        .insert(&pool)
        .await;
    SeedChunk::new(&page, "check", 2, "S_TCODE was checked by the old kernel.")
        .version("old-check")
        .superseded()
        .insert(&pool)
        .await;
    let check = SeedChunk::new(
        &page,
        "check",
        2,
        "S_TCODE is checked at transaction start.",
    )
    .version("new-check")
    .insert(&pool)
    .await;
    let gone = document(&pool, SourceKind::Markdown, "deleted.md", "Deleted file").await;
    let deleted = SeedChunk::new(&gone, "body", 0, "S_TCODE in a deleted file.")
        .superseded()
        .insert(&pool)
        .await;

    let results = retriever(&pool, HybridConfig::default())
        .retrieve("S_TCODE")
        .await
        .unwrap();

    // The hit pulls in the nearest current section, skipping the removed one.
    assert_eq!(ids(&results), [check.0.as_str(), intro.0.as_str()]);
    assert_eq!(results[0].chunk.version.0, "new-check");
    assert_eq!(results[1].chunk.version.0, "new-intro");
    assert_eq!(position(&results, &removed), None);
    assert_eq!(position(&results, &deleted), None);
}

#[sqlx::test(migrator = "drew::store::MIGRATOR")]
async fn hits_expand_to_neighbouring_sections(pool: PgPool) {
    let page = document(&pool, SourceKind::Markdown, "30", "SU24 guide").await;
    let mut sections = Vec::new();
    for (ordinal, text) in [
        "Overview of proposal values.",
        "Prerequisites for maintenance.",
        "Run SU24 and pick the application.",
        "Adjust check indicators.",
        "Transport the changes.",
    ]
    .into_iter()
    .enumerate()
    {
        let section = SeedChunk::new(&page, &format!("s{ordinal}"), ordinal as i32, text)
            .heading(&format!("Step {ordinal}"))
            .insert(&pool)
            .await;
        sections.push(section);
    }
    let sparse = document(&pool, SourceKind::Markdown, "31", "Sparse").await;
    let first = SeedChunk::new(&sparse, "a", 0, "Background.")
        .insert(&pool)
        .await;
    let middle = SeedChunk::new(&sparse, "b", 4, "SU24 in sparse pages.")
        .insert(&pool)
        .await;
    let last = SeedChunk::new(&sparse, "c", 9, "Appendix.")
        .insert(&pool)
        .await;

    let results = retriever(&pool, full_text_only())
        .retrieve("SU24")
        .await
        .unwrap();

    let hit = position(&results, &sections[2]).unwrap();
    let hit_score = results[hit].score;
    for neighbour in [&sections[1], &sections[3]] {
        let at = position(&results, neighbour).unwrap();
        assert_eq!(at, hit + if neighbour == &sections[1] { 1 } else { 2 });
        assert_eq!(results[at].score, Relevance(hit_score.0.next_down()));
    }
    for far in [&sections[0], &sections[4]] {
        assert_eq!(position(&results, far), None);
    }
    for id in [&first, &middle, &last] {
        assert!(position(&results, id).is_some(), "{id:?}");
    }
    assert_eq!(results.len(), 6);
}

#[sqlx::test(migrator = "drew::store::MIGRATOR")]
async fn expansion_dedupes_and_caps_per_page(pool: PgPool) {
    let page = document(&pool, SourceKind::Markdown, "40", "Start checks").await;
    let mut sections = Vec::new();
    for (ordinal, text) in [
        "Overview of the start check.",
        "Kernel behaviour.",
        "S_TCODE first, then S_TCODE again for each call.",
        "S_TCODE once, with extra context words here.",
        "Troubleshooting with ST01.",
        "Related notes.",
    ]
    .into_iter()
    .enumerate()
    {
        let section = SeedChunk::new(&page, &format!("s{ordinal}"), ordinal as i32, text)
            .insert(&pool)
            .await;
        sections.push(section);
    }

    // Both hits are kept; the better hit's neighbours fill the page up to the cap of three.
    let results = retriever(&pool, full_text_only())
        .retrieve("S_TCODE")
        .await
        .unwrap();
    assert_eq!(
        ids(&results),
        [&sections[2], &sections[1], &sections[3]].map(|id| id.0.as_str())
    );

    let roomy = HybridConfig {
        max_chunks_per_document: 5,
        ..full_text_only()
    };
    let results = retriever(&pool, roomy).retrieve("S_TCODE").await.unwrap();
    assert_eq!(
        ids(&results),
        [&sections[2], &sections[1], &sections[3], &sections[4]].map(|id| id.0.as_str())
    );
    assert!(results.windows(2).all(|w| w[0].score >= w[1].score));
}

#[sqlx::test(migrator = "drew::store::MIGRATOR")]
async fn adjacent_hits_expand_and_single_section_files_stand_alone(pool: PgPool) {
    let guide = document(
        &pool,
        SourceKind::Markdown,
        "import-fails.md",
        "Import fails",
    )
    .await;
    let symptom = SeedChunk::new(&guide, "symptom", 0, "Import fails on the role upload.")
        .insert(&pool)
        .await;
    let cause = SeedChunk::new(&guide, "cause", 1, "Cause: SU24 data was missing.")
        .insert(&pool)
        .await;
    let faq = document(&pool, SourceKind::Markdown, "su24-faq.md", "SU24 FAQ").await;
    let answer = SeedChunk::new(&faq, "body", 0, "Is SU24 transported? Yes.")
        .insert(&pool)
        .await;

    let results = retriever(&pool, full_text_only())
        .retrieve("SU24")
        .await
        .unwrap();
    let mut found = ids(&results);
    found.sort();
    let mut expected = [symptom.0.as_str(), cause.0.as_str(), answer.0.as_str()];
    expected.sort();
    assert_eq!(found, expected);

    let context_for = |id: &str| {
        results
            .iter()
            .find(|hit| hit.chunk.id.0 == id)
            .unwrap()
            .context_for
            .clone()
    };
    assert_eq!(context_for(&symptom.0), Some(cause.clone()));
    assert_eq!(context_for(&cause.0), None);
    assert_eq!(context_for(&answer.0), None);
}

#[sqlx::test(migrator = "drew::store::MIGRATOR")]
async fn chunks_are_fully_populated(pool: PgPool) {
    let page = document(&pool, SourceKind::Markdown, "50", "Role design").await;
    let id = SeedChunk::new(
        &page,
        "naming",
        0,
        "Role names use the Z_ prefix: Z_ARM_ADMIN.",
    )
    .heading("Naming")
    .insert(&pool)
    .await;

    let results = retriever(&pool, HybridConfig::default())
        .retrieve("Z_ARM_ADMIN")
        .await
        .unwrap();

    let chunk = &results[0].chunk;
    assert_eq!(chunk.id, id);
    assert_eq!(chunk.version.0, "v1");
    assert_eq!(chunk.document_id, page);
    assert_eq!(chunk.source, SourceKind::Markdown);
    assert_eq!(chunk.title, "Role design");
    assert_eq!(chunk.heading.as_deref(), Some("Naming"));
    assert_eq!(chunk.url, "https://example.test/markdown:50#naming");
    assert_eq!(chunk.labels.acl, ["group:arm"]);
    assert!(!chunk.authoritative);
}

#[sqlx::test(migrator = "drew::store::MIGRATOR")]
async fn mismatched_embedding_model_is_refused(pool: PgPool) {
    let titan = ModelRef::new("amazon.titan-embed-text-v2", "2");
    let doc = document(&pool, SourceKind::Markdown, "note-12.md", "Roles").await;
    SeedChunk::new(&doc, "old", 0, "AGR_USERS old vectors.")
        .model(titan.clone())
        .superseded()
        .insert(&pool)
        .await;
    SeedChunk::new(&doc, "body", 1, "AGR_USERS maps users to roles.")
        .insert(&pool)
        .await;
    let retriever = retriever(&pool, HybridConfig::default());
    assert_eq!(retriever.retrieve("AGR_USERS").await.unwrap().len(), 1);

    SeedChunk::new(&doc, "comment", 2, "AGR_USERS re-embedded with Titan.")
        .model(titan.clone())
        .insert(&pool)
        .await;
    let error = retriever.retrieve("AGR_USERS").await.unwrap_err();
    let mismatch = error
        .downcast_inner_ref::<EmbeddingModelMismatch>()
        .expect("a model mismatch");
    assert_eq!(mismatch.expected, HashEmbedder.model());
    assert_eq!(mismatch.found, [titan, HashEmbedder.model()]);
    assert!(
        error.to_string().contains("amazon.titan-embed-text-v2@2"),
        "{error}"
    );
}

#[sqlx::test(migrator = "drew::store::MIGRATOR")]
async fn empty_corpus_and_blank_query_return_nothing(pool: PgPool) {
    let retriever = retriever(&pool, HybridConfig::default());
    assert!(retriever.retrieve("AGR_USERS").await.unwrap().is_empty());
    let doc = document(&pool, SourceKind::Markdown, "note-13.md", "Roles").await;
    SeedChunk::new(&doc, "body", 0, "AGR_USERS")
        .insert(&pool)
        .await;
    assert!(retriever.retrieve("  ").await.unwrap().is_empty());
}

#[sqlx::test(migrator = "drew::store::MIGRATOR")]
async fn vector_side_uses_hnsw_beyond_its_default_ef_search(pool: PgPool) {
    for i in 0..60 {
        let doc = document(&pool, SourceKind::Markdown, &format!("V-{i}"), "filler").await;
        SeedChunk::new(&doc, "body", 0, &format!("note {i} on payroll"))
            .insert(&pool)
            .await;
    }
    let options = (*pool.connect_options())
        .clone()
        .options([("enable_seqscan", "off")]);
    let index_only = PgPoolOptions::new().connect_with(options).await.unwrap();

    let plan: Vec<String> = sqlx::query_scalar(
        "EXPLAIN SELECT id FROM chunks WHERE superseded_at IS NULL ORDER BY embedding <=> $1 LIMIT 50",
    )
    .bind(pgvector::Vector::from(embed("payroll").await))
    .fetch_all(&index_only)
    .await
    .unwrap();
    assert!(
        plan.iter()
            .any(|line| line.contains("chunks_current_embedding")),
        "{plan:?}"
    );

    let config = HybridConfig {
        full_text_candidates: 0,
        ..HybridConfig::default()
    };
    let results = retriever(&index_only, config)
        .retrieve("payroll")
        .await
        .unwrap();
    assert_eq!(results.len(), 50);
}

#[sqlx::test(migrator = "drew::store::MIGRATOR")]
async fn recall_at_k_on_fixture_set(pool: PgPool) {
    let chunk = async |source: SourceKind, key: &str, text: &str| {
        let doc = document(&pool, source, key, key).await;
        SeedChunk::new(&doc, "body", 0, text).insert(&pool).await
    };
    let start = chunk(
        SourceKind::Markdown,
        "60",
        "The kernel checks S_TCODE before any transaction starts.",
    )
    .await;
    let assignments = chunk(
        SourceKind::Markdown,
        "61",
        "AGR_USERS lists which users hold which roles, with validity dates.",
    )
    .await;
    let proposals = chunk(
        SourceKind::Markdown,
        "note-20.md",
        "SU24 proposal values feed PFCG when roles are built.",
    )
    .await;
    let tables = chunk(
        SourceKind::Markdown,
        "tables.md",
        "S_TABU_DIS protects table maintenance by authorization group.",
    )
    .await;
    let leavers = chunk(
        SourceKind::Markdown,
        "leavers.md",
        "Leavers are locked first, then their roles are removed.",
    )
    .await;
    for (i, text) in ["Payroll cluster tables.", "Spool settings for printers."]
        .into_iter()
        .enumerate()
    {
        chunk(SourceKind::Markdown, &format!("note-3{i}.md"), text).await;
    }
    let golden: [(&str, &[&ChunkId]); 5] = [
        ("Is S_TCODE checked at transaction start?", &[&start]),
        (
            "Where are role assignments for users stored?",
            &[&assignments],
        ),
        ("Does SU24 feed PFCG?", &[&proposals]),
        ("Which object protects table maintenance?", &[&tables]),
        ("What happens to leavers' roles?", &[&leavers, &assignments]),
    ];
    let retriever = retriever(&pool, HybridConfig::default());

    let mut recall = Vec::new();
    for (question, relevant) in golden {
        let results = retriever.retrieve(question).await.unwrap();
        let relevant: Vec<ChunkId> = relevant.iter().map(|id| (*id).clone()).collect();
        recall.push(recall_at_k(&results, &relevant, 5));
    }
    let mean = recall.iter().sum::<f64>() / recall.len() as f64;
    assert_eq!(mean, 1.0, "{recall:?}");

    let results = retriever.retrieve("Does SU24 feed PFCG?").await.unwrap();
    assert_eq!(
        recall_at_k(&results, std::slice::from_ref(&proposals), 1),
        1.0
    );
    assert_eq!(recall_at_k(&results, &[proposals, start], 1), 0.5);
}

#[sqlx::test(migrator = "drew::store::MIGRATOR")]
async fn a_glossary_synonym_retrieves_its_target(pool: PgPool) {
    let lexicon = document(&pool, SourceKind::Markdown, "LEXICON.md", "Lexicon").await;
    let row = SeedChunk::new(
        &lexicon,
        "generations",
        0,
        "| Internal | Say instead |\n|---|---|\n\
         | Legacy / NEW generation | \"the current screens\" and \"the earlier screens\" |",
    )
    .insert(&pool)
    .await;
    let doc = document(&pool, SourceKind::Markdown, "change.md", "Change").await;
    let target = SeedChunk::new(
        &doc,
        "body",
        0,
        "The earlier generation keeps its legacy behaviour until migration.",
    )
    .insert(&pool)
    .await;
    let question = "Is a fix on the NEW screen enough?";

    let plain = retriever(&pool, full_text_only())
        .retrieve(question)
        .await
        .unwrap();
    assert_eq!(position(&plain, &target), None);

    let config = HybridConfig {
        expansion: ExpansionConfig {
            documents: vec![lexicon],
            max_terms: 16,
        },
        ..full_text_only()
    };
    let expanded = retriever(&pool, config)
        .retrieve_expanded(question)
        .await
        .unwrap();
    assert!(position(&expanded.hits, &target).is_some());
    let matches = expanded.expansion.matches();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].matched, "new");
    assert_eq!(matches[0].source.id, row);
    assert_eq!(
        matches[0].terms,
        ["legacy", "generation", "current", "screens", "earlier"]
    );
}
