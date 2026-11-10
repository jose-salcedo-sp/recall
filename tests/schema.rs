//! Schema constraints: check constraints accept every domain enum variant, and a chunk has one current version.

#![allow(
    clippy::unwrap_used,
    reason = "integration-test helpers; clippy exempts only #[test] functions"
)]

use pgvector::Vector;
use recall::{
    domain::{CategoryKind, Sensitivity, SourceKind},
    llm::EMBEDDING_DIM,
};
use sqlx::PgPool;
use uuid::Uuid;

#[sqlx::test(migrator = "recall::store::MIGRATOR")]
async fn check_constraints_accept_every_enum_variant(pool: PgPool) -> sqlx::Result<()> {
    let document = |source: &str| {
        sqlx::query(
            "INSERT INTO documents (id, source, external_id, title, url, source_updated_at)
             VALUES ($1 || ':1', $1, '1', 'Roles', 'https://example.test/1', now())",
        )
        .bind(source.to_owned())
    };
    for source in SourceKind::ALL {
        document(source.as_str()).execute(&pool).await?;
    }
    for source in ["unknown", "wiki"] {
        assert!(document(source).execute(&pool).await.is_err(), "{source}");
    }

    let chunk = |locator: &str, sensitivity: &str| {
        sqlx::query(
            "INSERT INTO chunks (id, version, document_id, ordinal, url, text, authoritative, acl,
                                 sensitivity, embedding, embedding_model, embedding_model_version)
             VALUES ($1, 'v1', 'markdown:1', 0, 'https://example.test/1', 'text', true, '{}', $2, $3,
                     'fake', '1')",
        )
        .bind(format!("markdown:1#{locator}"))
        .bind(sensitivity.to_owned())
        .bind(Vector::from(vec![0.0; EMBEDDING_DIM]))
    };
    for sensitivity in Sensitivity::ALL {
        chunk(sensitivity.as_str(), sensitivity.as_str())
            .execute(&pool)
            .await?;
    }
    assert!(chunk("unknown", "unknown").execute(&pool).await.is_err());

    let run = |category: &str| {
        sqlx::query(
            "INSERT INTO pipeline_runs (id, user_id, question, category, status, outcome, trace)
             VALUES ($1, 'jose', 'q', $2, 'failed', '{}', '{}')",
        )
        .bind(Uuid::now_v7())
        .bind(category.to_owned())
    };
    for category in CategoryKind::ALL {
        run(category.as_str()).execute(&pool).await?;
    }
    assert!(run("unknown").execute(&pool).await.is_err());
    Ok(())
}

#[sqlx::test(migrator = "recall::store::MIGRATOR")]
async fn a_chunk_has_one_current_version(pool: PgPool) -> sqlx::Result<()> {
    sqlx::query(
        "INSERT INTO documents (id, source, external_id, title, url, source_updated_at)
         VALUES ('markdown:roles.md', 'markdown', 'roles.md', 'Roles', 'https://example.test/roles.md', now())",
    )
    .execute(&pool)
    .await?;

    let insert = |version: &'static str| {
        sqlx::query(
            "INSERT INTO chunks (id, version, document_id, ordinal, url, text, authoritative, acl,
                                 sensitivity, embedding, embedding_model, embedding_model_version)
             VALUES ('markdown:roles.md#intro', $1, 'markdown:roles.md', 0, 'https://example.test/roles.md',
                     'text', true, '{}', 'internal', $2, 'fake', '1')",
        )
        .bind(version)
        .bind(Vector::from(vec![0.0; EMBEDDING_DIM]))
    };

    insert("v1").execute(&pool).await?;
    assert!(insert("v2").execute(&pool).await.is_err());

    sqlx::query("UPDATE chunks SET superseded_at = now() WHERE version = 'v1'")
        .execute(&pool)
        .await?;
    insert("v2").execute(&pool).await?;
    Ok(())
}
