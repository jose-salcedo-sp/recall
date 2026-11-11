//! Options shared by several commands: the database, the pipeline config file and the access policy file.

use std::path::{Path, PathBuf};

use sqlx::PgPool;

use crate::{
    access::rules::{AccessConfig, DEFAULT_PATH as ACCESS_PATH},
    config::{DEFAULT_PATH as CONFIG_PATH, PipelineConfig},
    store,
};

/// Database connection options.
#[derive(Clone, Debug, clap::Args)]
pub struct DbArgs {
    /// Postgres connection URL.
    #[arg(long, env = "DATABASE_URL")]
    pub database_url: String,
}

impl DbArgs {
    /// Connects and applies pending migrations.
    ///
    /// # Errors
    ///
    /// As [`store::connect`].
    pub async fn connect(&self) -> eros::Result<PgPool> {
        store::connect(&self.database_url).await
    }
}

/// Where to load the [`PipelineConfig`] from.
#[derive(Clone, Debug, Default, clap::Args)]
pub struct ConfigArgs {
    /// Pipeline configuration JSON. Unset: `config/recall.json` when it exists, else built-in defaults.
    #[arg(long = "config", env = "RECALL_CONFIG")]
    pub path: Option<PathBuf>,
}

impl ConfigArgs {
    /// Loads and validates the config. An empty `RECALL_CONFIG`, as in `.env.example`, counts as unset.
    ///
    /// # Errors
    ///
    /// When the file cannot be read, is not valid JSON, has unknown keys, or fails [`PipelineConfig::parse`]'s checks.
    pub fn load(&self) -> eros::Result<PipelineConfig> {
        match self
            .path
            .as_ref()
            .filter(|path| !path.as_os_str().is_empty())
        {
            Some(path) => PipelineConfig::read(path),
            None if Path::new(CONFIG_PATH).exists() => PipelineConfig::read(Path::new(CONFIG_PATH)),
            None => {
                tracing::warn!("no {CONFIG_PATH} and no --config; using built-in defaults");
                Ok(PipelineConfig::default())
            }
        }
    }
}

/// Where to load the [`AccessConfig`] from.
#[derive(Clone, Debug, clap::Args)]
pub struct AccessArgs {
    /// Access policy JSON: roles, members and label rules. Required: without it nobody may read anything.
    #[arg(id = "access", long = "access", env = "RECALL_ACCESS", default_value = ACCESS_PATH)]
    pub path: PathBuf,
}

impl AccessArgs {
    /// Reads and validates the policy.
    ///
    /// # Errors
    ///
    /// As [`AccessConfig::read`].
    pub fn load(&self) -> eros::Result<AccessConfig> {
        AccessConfig::read(&self.path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_config_path_is_unset_and_a_missing_one_is_an_error() {
        let empty = ConfigArgs {
            path: Some(PathBuf::new()),
        };
        assert!(empty.load().is_ok());
        let missing = ConfigArgs {
            path: Some("no/such/recall.json".into()),
        };
        assert!(missing.load().is_err());
    }
}
