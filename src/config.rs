use std::time::Duration;

/// Every knob from plan.md's "Revised knobs" table, read from env once at startup.
#[derive(Debug, Clone)]
pub struct Config {
    pub bind_addr: String,
    /// Nexus Supabase, **session** pooler, as role `recall_search`. That role can
    /// execute the two search functions and nothing else — no table reads — so the
    /// corpus stays Nexus-owned and Recall cannot widen its own access.
    pub database_url: String,
    pub embedder_url: String,
    /// Must match what Nexus indexed with, or the vectors are not comparable.
    pub embedding_model: String,
    pub embedding_dim: usize,
    pub system_one_url: String,
    pub generator_url: String,
    pub generator_model: String,

    /// Bearer token Nexus presents. No end-user JWT ever reaches Recall.
    pub service_token: Option<String>,
    /// Writing the ask record needs a table Recall does not own on Nexus. Off by
    /// default so the audit write cannot fail every request on that deployment.
    pub persist_asks: bool,
    /// The index write endpoints only make sense against a Recall-owned corpus.
    pub index_writes_enabled: bool,

    pub retrieve_k: i32,
    pub admit_batch_size: usize,
    /// Deliberately has no default worth trusting. See plan.md Finding 4: until a
    /// temperature is fitted on labeled pairs, any threshold here is a guess.
    pub admit_threshold: f64,
    pub admit_threshold_is_calibrated: bool,
    pub max_citations: usize,
    pub max_inflight: usize,

    pub embed_timeout: Duration,
    pub index_timeout: Duration,
    pub admit_timeout: Duration,
    pub ask_timeout: Duration,
    pub stage_ping_interval: Duration,
}

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_str(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

impl Config {
    pub fn from_env() -> Self {
        Self {
            bind_addr: env_str("BIND_ADDR", "0.0.0.0:8000"),
            // No default: pointing at the wrong database is the one misconfiguration
            // that could silently answer from an empty or foreign corpus.
            database_url: env_str("DATABASE_URL", ""),
            embedder_url: env_str("EMBEDDER_URL", "https://api.openai.com"),
            embedding_model: env_str("NEXUS_EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dim: env_or("EMBEDDING_DIM", 1536),
            system_one_url: env_str("SYSTEM_ONE_URL", "http://localhost:8082"),
            generator_url: env_str("GENERATOR_URL", "http://localhost:8080"),
            generator_model: env_str("GENERATOR_MODEL", "local"),

            service_token: std::env::var("RECALL_SERVICE_TOKEN")
                .ok()
                .filter(|s| !s.trim().is_empty()),
            persist_asks: env_or("PERSIST_ASKS", false),
            index_writes_enabled: env_or("INDEX_WRITES_ENABLED", false),

            retrieve_k: env_or("RETRIEVE_K", 32),
            admit_batch_size: env_or("ADMIT_BATCH_SIZE", 32),
            // Provisional: measured on four probes, not fitted on a labeled set.
            // Answerable questions topped out at 0.261-0.665 and an unanswerable one
            // at 0.045, so anything in ~0.1-0.25 separates them; architecture.md's
            // 0.7 would refuse all of them. Phase 4 replaces this with a fitted
            // value and a precision/recall curve. See plan.md Finding 4.
            admit_threshold: env_or("ADMIT_THRESHOLD", 0.15),
            admit_threshold_is_calibrated: env_or("ADMIT_THRESHOLD_CALIBRATED", false),
            max_citations: env_or("MAX_CITATIONS", 4),
            max_inflight: env_or("MAX_INFLIGHT", 4),

            embed_timeout: Duration::from_millis(env_or("EMBED_TIMEOUT_MS", 5_000)),
            index_timeout: Duration::from_millis(env_or("INDEX_TIMEOUT_MS", 5_000)),
            admit_timeout: Duration::from_millis(env_or("ADMIT_TIMEOUT_MS", 30_000)),
            ask_timeout: Duration::from_millis(env_or("ASK_TIMEOUT_MS", 60_000)),
            stage_ping_interval: Duration::from_millis(env_or("STAGE_PING_INTERVAL_MS", 15_000)),
        }
    }

    /// Refuse to start rather than run in a shape that could leak or mislead.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.database_url.trim().is_empty() {
            anyhow::bail!("DATABASE_URL is required (Nexus session pooler, role recall_search)");
        }
        if self.service_token.is_none() {
            anyhow::bail!(
                "RECALL_SERVICE_TOKEN is required: /v1/ask* and /v1/asks/* must not be \
                 reachable unauthenticated. Set it to the token issued to Nexus."
            );
        }
        // Checking only that sslmode is *present* would accept sslmode=disable, so
        // check the value. The escape hatch is named to be uncomfortable to set in
        // anything but the local Compose stack, whose Postgres has no TLS.
        let tls_ok = ["sslmode=require", "sslmode=verify-ca", "sslmode=verify-full"]
            .iter()
            .any(|m| self.database_url.contains(m));
        if !tls_ok && !env_or("ALLOW_INSECURE_DB", false) {
            anyhow::bail!(
                "DATABASE_URL must set sslmode=require (or verify-ca/verify-full) so the \
                 pooler connection cannot fall back to plaintext. For the local Compose \
                 Postgres only, set ALLOW_INSECURE_DB=true."
            );
        }
        if !tls_ok {
            tracing::warn!(
                "ALLOW_INSECURE_DB is set: the database connection is not using TLS. \
                 This must never be set against Nexus."
            );
        }
        if self.database_url.contains("service_role") {
            anyhow::bail!("refusing to run as service_role; use the recall_search role");
        }
        Ok(())
    }

    /// Loud on startup rather than silently wrong in production.
    pub fn warn_if_uncalibrated(&self) {
        if !self.admit_threshold_is_calibrated {
            tracing::warn!(
                threshold = self.admit_threshold,
                "ADMIT_THRESHOLD is not calibrated: it is a placeholder, not a fitted value. \
                 Admission precision is unknown until a temperature is fitted on labeled \
                 (question, candidate) pairs. See plan.md Phase 4."
            );
        }
    }
}
