use std::time::Duration;

/// Knobs from plan.md. Local admission is CPU Laya; fitted cutoffs transfer,
/// `max_inflight` does not.
#[derive(Debug, Clone)]
pub struct Config {
    pub bind_addr: String,
    pub database_url: String,
    pub embedder_url: String,
    pub embedding_model: String,
    pub embedding_dim: usize,
    pub system_one_url: String,
    pub generator_url: String,
    pub generator_model: String,

    pub service_token: Option<String>,
    pub persist_asks: bool,
    pub index_writes_enabled: bool,

    pub retrieve_k: i32,
    pub admit_batch_size: usize,
    pub injection_max: f64,
    pub contradicts_min: f64,
    pub relevant_min: f64,
    pub evidence_min: f64,
    pub admit_thresholds_calibrated: bool,
    pub max_citations: usize,
    pub max_inflight: usize,
    pub kind_confidence_min: f64,
    pub cite_confidence_min: f64,
    pub verify_regen_max: u32,

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

            retrieve_k: env_or("RETRIEVE_K", 64),
            admit_batch_size: env_or("ADMIT_BATCH_SIZE", 32),
            // Provisional local-Laya cutoffs until phase 6 fits a PR curve.
            injection_max: env_or("INJECTION_MAX", 0.70),
            contradicts_min: env_or("CONTRADICTS_MIN", 0.70),
            relevant_min: env_or("RELEVANT_MIN", 0.10),
            evidence_min: env_or("EVIDENCE_MIN", 0.10),
            admit_thresholds_calibrated: env_or("ADMIT_THRESHOLDS_CALIBRATED", false),
            max_citations: env_or("MAX_CITATIONS", 4),
            max_inflight: env_or("MAX_INFLIGHT", 4),
            kind_confidence_min: env_or("KIND_CONFIDENCE_MIN", 0.8),
            // P(chosen relate class). 0.75 keeps a clear paraphrase (~0.79) and
            // drops a weak argmax (~0.46). Laya's entropy "confidence" is not this number.
            cite_confidence_min: env_or("CITE_CONFIDENCE_MIN", 0.75),
            verify_regen_max: env_or("VERIFY_REGEN_MAX", 1),

            embed_timeout: Duration::from_millis(env_or("EMBED_TIMEOUT_MS", 5_000)),
            index_timeout: Duration::from_millis(env_or("INDEX_TIMEOUT_MS", 5_000)),
            admit_timeout: Duration::from_millis(env_or("ADMIT_TIMEOUT_MS", 50_000)),
            ask_timeout: Duration::from_millis(env_or("ASK_TIMEOUT_MS", 60_000)),
            stage_ping_interval: Duration::from_millis(env_or("STAGE_PING_INTERVAL_MS", 15_000)),
        }
    }

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
        let tls_ok = [
            "sslmode=require",
            "sslmode=verify-ca",
            "sslmode=verify-full",
        ]
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

    pub fn warn_if_uncalibrated(&self) {
        if !self.admit_thresholds_calibrated {
            tracing::warn!(
                injection_max = self.injection_max,
                contradicts_min = self.contradicts_min,
                relevant_min = self.relevant_min,
                evidence_min = self.evidence_min,
                "ADMIT_THRESHOLDS_CALIBRATED is false: filter cutoffs are cookbook \
                 placeholders, not a fitted PR curve. See plan.md Phase 6."
            );
        }
    }
}
