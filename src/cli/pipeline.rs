//! The pipeline as `recall serve`, `recall eval run` and `recall models check` configure it.

use std::sync::Arc;

use sqlx::PgPool;

use crate::{
    access::rules::RulePolicy,
    clock::{SystemClock, UuidV7},
    pipeline::{answerer::PipelineAnswerer, assembly::Ports},
    retrieval::hybrid::HybridRetriever,
};

use super::{
    args::{AccessArgs, ConfigArgs},
    backend::{self, ChatArgs, EmbedArgs, SystemOneArgs},
};

/// Everything [`PipelineArgs::answerer`] needs besides the database.
#[derive(Clone, Debug, clap::Args)]
pub struct PipelineArgs {
    /// The tunables file.
    #[command(flatten)]
    pub config: ConfigArgs,
    /// The query embedder; must match the corpus's model.
    #[command(flatten)]
    pub embed: EmbedArgs,
    /// The `Explain` LLM.
    #[command(flatten)]
    pub chat: ChatArgs,
    /// The judge's backend (System One), for `Classify`, `Answer`, and `Rerank`, `Admit` and `Verify` when their
    /// config names a judge.
    #[command(flatten)]
    pub system_one: SystemOneArgs,
    /// The access policy every run is authorized by.
    #[command(flatten)]
    pub access: AccessArgs,
}

impl PipelineArgs {
    /// Hybrid retrieval over `pool` and `Explain` on Bedrock or OpenAI-compatible servers; `Classify` and `Answer`
    /// on the judge when one is configured, and `Rerank`, `Admit` and `Verify`'s support check too when their config
    /// names it. Every tunable comes from the pipeline config file.
    ///
    /// # Errors
    ///
    /// When the config or the access policy does not load, a model client cannot be built, or the config does not
    /// assemble with the backends selected.
    pub async fn answerer(&self, pool: PgPool) -> eros::Result<PipelineAnswerer> {
        let config = self.config.load()?;
        let authorizer = Arc::new(RulePolicy::new(self.access.load()?));
        let embedder = backend::embedder(&self.embed).await?;
        let embedding = Some(embedder.model());
        let retriever = HybridRetriever::new(pool, embedder, config.retrieval.clone());
        let ports = Ports {
            retriever: Arc::new(retriever),
            embedding,
            explain: backend::explain_model(&self.chat).await?,
            judge: backend::judge(&self.system_one)?,
            authorizer,
            clock: Arc::new(SystemClock),
            ids: Arc::new(UuidV7),
        };
        Ok(PipelineAnswerer::assemble(config, ports)?)
    }
}
