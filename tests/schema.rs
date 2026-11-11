//! Schema constraints: check constraints accept every domain enum variant, exactly the valid source names and ACL
//! entries, and a chunk has one current version; migrations rewrite stored data into the current formats.

#![allow(
    clippy::unwrap_used,
    reason = "integration-test helpers; clippy exempts only #[test] functions"
)]

use pgvector::Vector;
use recall::{
    access::AclEntry,
    domain::{CategoryKind, Sensitivity, SourceName},
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
    for source in [
        "markdown",
        "confluence",
        "jira-cloud",
        "db_2",
        "",
        "Markdown",
        "wiki:pages",
        "a b",
        "doc#1",
        "café",
    ] {
        assert_eq!(
            document(source).execute(&pool).await.is_ok(),
            SourceName::new(source).is_ok(),
            "{source:?}"
        );
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
            "INSERT INTO pipeline_runs (id, user_id, question, category, outcome, trace)
             VALUES ($1, 'jose', 'q', $2, '{\"status\": \"failed\"}', '{}')",
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

#[sqlx::test(migrator = "recall::store::MIGRATOR")]
async fn the_acl_check_accepts_exactly_the_entries_acl_entry_parses(
    pool: PgPool,
) -> sqlx::Result<()> {
    sqlx::query(
        "INSERT INTO documents (id, source, external_id, title, url, source_updated_at)
         VALUES ('markdown:1', 'markdown', '1', 'Roles', 'https://example.test/1', now())",
    )
    .execute(&pool)
    .await?;
    for (ordinal, entry) in [
        "user:ana",
        "user:Ana Smith",
        "role:hr",
        "role:hr:payroll",
        "hr",
        "group:hr",
        "user:",
        "role:",
        "role:HR",
        "user:a\nb",
    ]
    .into_iter()
    .enumerate()
    {
        let inserted = sqlx::query(
            "INSERT INTO chunks (id, version, document_id, ordinal, url, text, authoritative, acl,
                                 sensitivity, embedding, embedding_model, embedding_model_version)
             VALUES ($1, 'v1', 'markdown:1', $2, 'https://example.test/1', 'text', true, ARRAY[$3], 'internal',
                     $4, 'fake', '1')",
        )
        .bind(format!("markdown:1#{ordinal}"))
        .bind(i32::try_from(ordinal).unwrap())
        .bind(entry)
        .bind(Vector::from(vec![0.0; EMBEDDING_DIM]))
        .execute(&pool)
        .await;
        assert_eq!(
            inserted.is_ok(),
            entry.parse::<AclEntry>().is_ok(),
            "{entry:?}"
        );
    }
    Ok(())
}

const TAGGED_ACL_MIGRATION: i64 = 20261111000000;

/// Applies every migration before `version`, then runs `seed`, then applies the rest.
async fn migrate_around(pool: &PgPool, version: i64, seed: String) -> sqlx::Result<()> {
    for migration in recall::store::MIGRATOR.iter() {
        if migration.version == version {
            sqlx::raw_sql(sqlx::AssertSqlSafe(seed.clone()))
                .execute(pool)
                .await?;
        }
        sqlx::raw_sql(migration.sql.clone()).execute(pool).await?;
    }
    Ok(())
}

/// Untagged ACL entries and stored scope members become tagged, never naming more than they did.
#[sqlx::test(migrations = false)]
async fn the_tagged_acl_migration_rewrites_stored_entries(pool: PgPool) -> sqlx::Result<()> {
    let trace = serde_json::json!([
        {
            "step": "authorize",
            "record": {
                "kind": "authorize",
                "access": { "chunks": { "members": ["hr", "reader"], "grants": [] }, "runs": { "runs": "nothing" } }
            }
        },
        { "step": "retrieve", "record": null }
    ]);
    let seed = format!(
        "INSERT INTO documents (id, source, external_id, title, url, source_updated_at)
         VALUES ('markdown:1', 'markdown', '1', 'Roles', 'https://example.test/1', now());
         INSERT INTO chunks (id, version, document_id, ordinal, url, text, authoritative, acl, sensitivity,
                             embedding, embedding_model, embedding_model_version)
         VALUES ('markdown:1#a', 'v1', 'markdown:1', 0, 'u', 't', true, ARRAY['hr', 'Ana Smith'], 'internal',
                 array_fill(0, ARRAY[{EMBEDDING_DIM}])::vector, 'fake', '1'),
                ('markdown:1#b', 'v1', 'markdown:1', 1, 'u', 't', true, '{{}}', 'internal',
                 array_fill(0, ARRAY[{EMBEDDING_DIM}])::vector, 'fake', '1');
         INSERT INTO pipeline_runs (id, user_id, question, status, outcome, trace)
         VALUES ('{}', 'hr', 'q', 'failed', '{{\"status\": \"failed\"}}', '{}');",
        Uuid::now_v7(),
        trace
    );

    migrate_around(&pool, TAGGED_ACL_MIGRATION, seed).await?;

    let acls: Vec<Vec<String>> = sqlx::query_scalar("SELECT acl FROM chunks ORDER BY id")
        .fetch_all(&pool)
        .await?;
    assert_eq!(
        acls,
        [
            vec!["role:hr".to_owned(), "user:Ana Smith".to_owned()],
            vec![]
        ]
    );
    let trace: serde_json::Value = sqlx::query_scalar("SELECT trace FROM pipeline_runs")
        .fetch_one(&pool)
        .await?;
    assert_eq!(
        trace[0]["record"]["access"]["chunks"]["members"],
        serde_json::json!(["role:reader", "user:hr"])
    );
    assert_eq!(trace[1]["step"], "retrieve");
    Ok(())
}

const SINGLE_TENANT_MIGRATION: i64 = 20261111010000;

/// A document with one chunk of tenant `acme`, current or superseded.
fn tenanted_chunk(superseded: bool) -> String {
    format!(
        "INSERT INTO documents (id, source, external_id, title, url, source_updated_at)
         VALUES ('markdown:1', 'markdown', '1', 'Roles', 'https://example.test/1', now());
         INSERT INTO chunks (id, version, document_id, ordinal, url, text, authoritative, acl, tenant, sensitivity,
                             embedding, embedding_model, embedding_model_version, superseded_at)
         VALUES ('markdown:1#a', 'v1', 'markdown:1', 0, 'u', 't', true, '{{}}', 'acme', 'internal',
                 array_fill(0, ARRAY[{EMBEDDING_DIM}])::vector, 'fake', '1', {});",
        if superseded { "now()" } else { "NULL" }
    )
}

/// Dropping the tenant of a current chunk would widen who reads it, so the migration refuses.
#[sqlx::test(migrations = false)]
async fn the_single_tenant_migration_refuses_while_a_current_chunk_has_a_tenant(
    pool: PgPool,
) -> sqlx::Result<()> {
    let error = migrate_around(&pool, SINGLE_TENANT_MIGRATION, tenanted_chunk(false))
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("current chunks carry a tenant"),
        "{error}"
    );
    Ok(())
}

#[sqlx::test(migrations = false)]
async fn the_single_tenant_migration_drops_the_column(pool: PgPool) -> sqlx::Result<()> {
    migrate_around(&pool, SINGLE_TENANT_MIGRATION, tenanted_chunk(true)).await?;
    let columns: Vec<String> = sqlx::query_scalar(
        "SELECT column_name::text FROM information_schema.columns WHERE table_name = 'chunks'",
    )
    .fetch_all(&pool)
    .await?;
    assert!(
        !columns.iter().any(|column| column == "tenant"),
        "{columns:?}"
    );
    Ok(())
}

/// `status` is derived from the outcome's tag, so the two cannot disagree; an unknown tag does not store.
#[sqlx::test(migrator = "recall::store::MIGRATOR")]
async fn the_status_column_is_the_outcome_tag(pool: PgPool) -> sqlx::Result<()> {
    let insert = |outcome: serde_json::Value| {
        sqlx::query(
            "INSERT INTO pipeline_runs (id, user_id, question, outcome, trace) VALUES ($1, 'jose', 'q', $2, '[]')
             RETURNING status",
        )
        .bind(Uuid::now_v7())
        .bind(outcome)
    };
    for status in ["answered", "declined", "failed"] {
        let stored: String = insert(serde_json::json!({ "status": status }))
            .fetch_one(&pool)
            .await
            .map(|row| sqlx::Row::get(&row, "status"))?;
        assert_eq!(stored, status);
    }
    assert!(
        insert(serde_json::json!({ "status": "pending" }))
            .execute(&pool)
            .await
            .is_err()
    );
    let written = sqlx::query(
        "INSERT INTO pipeline_runs (id, user_id, question, status, outcome, trace)
         VALUES ($1, 'jose', 'q', 'failed', '{\"status\": \"answered\"}', '[]')",
    )
    .bind(Uuid::now_v7())
    .execute(&pool)
    .await;
    assert!(written.is_err(), "status is generated, never written");
    Ok(())
}

const EXPLAIN_BY_PATH_MIGRATION: i64 = 20261111030000;

/// An open-path explain record loses the flag it never meant anything on; a verdict-path one keeps it.
#[sqlx::test(migrations = false)]
async fn the_explain_migration_keeps_the_flag_only_on_the_verdict_path(
    pool: PgPool,
) -> sqlx::Result<()> {
    let explain = serde_json::json!({ "step": "explain", "record": { "kind": "explain", "raw": "{}", "contradicts_verdict": false } });
    let answer = serde_json::json!({ "step": "answer", "record": { "kind": "answer", "raw": null, "parsed": null } });
    let open = serde_json::json!([explain]);
    let verdict = serde_json::json!([answer, explain]);
    let seed = format!(
        "INSERT INTO pipeline_runs (id, user_id, question, outcome, trace) VALUES
         ('{}', 'ana', 'open', '{{\"status\": \"failed\"}}', '{open}'),
         ('{}', 'ana', 'verdict', '{{\"status\": \"failed\"}}', '{verdict}');",
        Uuid::now_v7(),
        Uuid::now_v7(),
    );

    migrate_around(&pool, EXPLAIN_BY_PATH_MIGRATION, seed).await?;

    let traces: Vec<(String, serde_json::Value)> =
        sqlx::query_as("SELECT question, trace FROM pipeline_runs ORDER BY question")
            .fetch_all(&pool)
            .await?;
    assert_eq!(traces[0].0, "open");
    assert_eq!(
        traces[0].1[0]["record"],
        serde_json::json!({ "kind": "explain", "raw": "{}" })
    );
    assert_eq!(traces[1].1[1]["record"]["contradicts_verdict"], false);
    Ok(())
}
