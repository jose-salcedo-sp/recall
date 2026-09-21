use std::time::Duration;

/// Every knob from plan.md's "Revised knobs" table, read from env once at startup.
#[derive(Debug, Clone)]
pub struct Config {
    pub bind_addr: String,
    pub database_url: String,
    pub embedder_url: String,
    pub system_one_url: String,
    pub generator_url: String,
    pub generator_model: String,

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
            database_url: env_str(
                "DATABASE_URL",
                "postgresql://recall:recall@localhost:5432/recall",
            ),
            embedder_url: env_str("EMBEDDER_URL", "http://localhost:8081"),
            system_one_url: env_str("SYSTEM_ONE_URL", "http://localhost:8082"),
            generator_url: env_str("GENERATOR_URL", "http://localhost:8080"),
            generator_model: env_str("GENERATOR_MODEL", "local"),

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
