//! Every pipeline tunable in one JSON file (`config/drew.json`), so `drew eval` can print and record what it ran with.

use std::{
    collections::BTreeMap,
    fmt, fs,
    path::{Path, PathBuf},
    str::FromStr,
};

use serde::{Deserialize, Serialize};

use crate::{
    domain::{CategoryKind, Probability},
    retrieval::hybrid::HybridConfig,
};

/// Read when neither `--config` nor `DREW_CONFIG` names a file; relative to the working directory.
pub const DEFAULT_PATH: &str = "config/drew.json";

/// A step whose score `Run::then` checks against a threshold. `classify` has `classify.min_confidence` instead,
/// which falls back to `open` rather than declining; `explain` and `verify` are not scored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(into = "&'static str", try_from = "String")]
pub enum StepName {
    /// [`crate::pipeline::steps::Rerank`], scored by the best kept match's probability.
    Rerank,
    /// [`crate::pipeline::steps::Answer`], scored by the verdict's confidence.
    Answer,
}

impl StepName {
    /// Every variant, in declaration order.
    pub const ALL: [Self; 2] = [Self::Rerank, Self::Answer];

    /// The step's `Step::NAME`, as recorded in the trace.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rerank => "rerank",
            Self::Answer => "answer",
        }
    }
}

impl fmt::Display for StepName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<StepName> for &'static str {
    fn from(step: StepName) -> Self {
        step.as_str()
    }
}

/// A step name with no threshold, rejected when parsing a config or tune candidate.
#[derive(Debug, thiserror::Error)]
#[error(
    "no threshold applies to step `{0}`; thresholded steps are {steps} (classify uses classify.min_confidence)",
    steps = StepName::ALL.map(StepName::as_str).join(", ")
)]
pub struct UnknownStep(String);

impl FromStr for StepName {
    type Err = UnknownStep;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|step| step.as_str() == name)
            .ok_or_else(|| UnknownStep(name.to_owned()))
    }
}

impl TryFrom<String> for StepName {
    type Error = UnknownStep;

    fn try_from(name: String) -> Result<Self, Self::Error> {
        name.parse()
    }
}

/// Minimum score per thresholded step. A step not listed has no threshold.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Thresholds(BTreeMap<StepName, Probability>);

impl Thresholds {
    /// `None` when the step is not listed, so it has no threshold.
    pub fn get(&self, step: StepName) -> Option<Probability> {
        self.0.get(&step).copied()
    }

    /// No step is listed.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl FromIterator<(StepName, Probability)> for Thresholds {
    fn from_iter<T: IntoIterator<Item = (StepName, Probability)>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

/// Where to load the [`PipelineConfig`] from.
#[derive(Clone, Debug, Default, clap::Args)]
pub struct ConfigArgs {
    /// Pipeline configuration JSON. Unset: `config/drew.json` when it exists, else built-in defaults.
    #[arg(long = "config", env = "DREW_CONFIG")]
    pub path: Option<PathBuf>,
}

impl ConfigArgs {
    /// Loads and validates the config. An empty `DREW_CONFIG`, as in `.env.example`, counts as unset.
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
            None if Path::new(DEFAULT_PATH).exists() => {
                PipelineConfig::read(Path::new(DEFAULT_PATH))
            }
            None => {
                tracing::warn!("no {DEFAULT_PATH} and no --config; using built-in defaults");
                Ok(PipelineConfig::default())
            }
        }
    }
}

/// Every pipeline tunable. Missing keys take their defaults; unknown keys are rejected.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PipelineConfig {
    /// Hybrid search tunables.
    pub retrieval: HybridConfig,
    /// When a classification is confident enough to leave the `open` path.
    pub classify: ClassifyConfig,
    /// Who judges the matches, and how many are kept.
    pub rerank: RerankConfig,
    /// The "why" LLM's token budget.
    pub explain: ExplainConfig,
    /// The shape of a `drew eval tune` candidate's `thresholds`.
    pub thresholds: Thresholds,
    /// Overrides `thresholds` for runs classified into that category.
    pub per_category: PerCategory,
}

/// [`PipelineConfig::classify`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ClassifyConfig {
    /// Below this probability, a `boolean`, `choice` or `ranking` classification falls back to `open`.
    pub min_confidence: Option<Probability>,
}

/// [`PipelineConfig::explain`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExplainConfig {
    /// Upper bound on the tokens Explain's model generates, reasoning included on a reasoning model.
    pub max_tokens: MaxTokens,
}

/// Explain's chat `max_tokens`, in `1..=`[`MaxTokens::MAX`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(into = "u32", try_from = "u32")]
pub struct MaxTokens(u32);

impl MaxTokens {
    /// Enough for Explain's answer on a model that does not reason, such as `qwen3:14b` in Ollama.
    pub const DEFAULT: Self = Self(2048);
    /// The largest accepted budget. Above it a value is a typo, and no Explain model answers in more.
    pub const MAX: u32 = 65_536;

    /// # Errors
    ///
    /// When `tokens` is 0 or over [`MaxTokens::MAX`].
    pub fn new(tokens: u32) -> Result<Self, InvalidMaxTokens> {
        if (1..=Self::MAX).contains(&tokens) {
            Ok(Self(tokens))
        } else {
            Err(InvalidMaxTokens(tokens))
        }
    }

    /// The budget.
    pub fn get(self) -> u32 {
        self.0
    }
}

impl Default for MaxTokens {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl From<MaxTokens> for u32 {
    fn from(tokens: MaxTokens) -> Self {
        tokens.0
    }
}

impl TryFrom<u32> for MaxTokens {
    type Error = InvalidMaxTokens;

    fn try_from(tokens: u32) -> Result<Self, Self::Error> {
        Self::new(tokens)
    }
}

/// A `max_tokens` outside `1..=`[`MaxTokens::MAX`].
#[derive(Debug, thiserror::Error)]
#[error("explain.max_tokens must be between 1 and {max}, got {0}", max = MaxTokens::MAX)]
pub struct InvalidMaxTokens(u32);

/// [`PipelineConfig::rerank`]: `0 < top_k`, and `top_k <= candidates` when System One judges.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "RawRerankConfig")]
pub struct RerankConfig {
    top_k: usize,
    judge: RerankJudge,
    evidence_rank_penalty: usize,
}

/// Who judges the matches `Rerank` orders. Classify and Answer use System One either way.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RerankJudge {
    /// No model: matches keep retrieval order, with the evidence penalty, and the step is unscored.
    #[default]
    None,
    /// System One judges the best matches by fused score, when System One is configured.
    SystemOne(SystemOneJudge),
}

/// [`RerankJudge::SystemOne`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SystemOneJudge {
    /// Best matches by fused retrieval score that System One judges.
    pub candidates: usize,
}

/// A rerank config that breaks [`RerankConfig`]'s invariants.
#[derive(Debug, thiserror::Error)]
pub enum InvalidRerankConfig {
    /// `top_k` is 0.
    #[error("rerank.top_k must be at least 1")]
    NothingKept,
    /// System One judges fewer matches than are kept.
    #[error("rerank.top_k ({top_k}) exceeds the {candidates} candidates System One judges")]
    MoreKeptThanJudged {
        /// Matches kept.
        top_k: usize,
        /// Matches judged.
        candidates: usize,
    },
    /// The flat `candidates` of configs written before `judge`.
    #[error(
        r#"rerank.candidates applies only to the System One judge; write "judge": {{"system_one": {{"candidates": {0}}}}}, or "judge": "none""#
    )]
    CandidatesWithoutJudge(usize),
    /// `"judge": "system_one"`, which leaves out how many matches it judges.
    #[error(
        r#"rerank.judge "system_one" needs its candidates; write "judge": {{"system_one": {{"candidates": 25}}}}"#
    )]
    SystemOneWithoutCandidates,
    /// Neither `"none"` nor `{"system_one": {"candidates": N}}`.
    #[error("rerank.judge: {0}")]
    Judge(serde_json::Error),
}

impl RerankConfig {
    /// # Errors
    ///
    /// When `top_k` is 0, or exceeds the System One judge's `candidates`.
    pub fn new(
        top_k: usize,
        judge: RerankJudge,
        evidence_rank_penalty: usize,
    ) -> Result<Self, InvalidRerankConfig> {
        if top_k == 0 {
            return Err(InvalidRerankConfig::NothingKept);
        }
        match judge {
            RerankJudge::SystemOne(SystemOneJudge { candidates }) if candidates < top_k => {
                Err(InvalidRerankConfig::MoreKeptThanJudged { top_k, candidates })
            }
            RerankJudge::None | RerankJudge::SystemOne(_) => Ok(Self {
                top_k,
                judge,
                evidence_rank_penalty,
            }),
        }
    }

    /// Matches kept for `Answer` and `Explain`.
    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Who judges the matches.
    pub fn judge(&self) -> RerankJudge {
        self.judge
    }

    /// An evidence (non-authoritative) match is ranked as if this many places lower.
    pub fn evidence_rank_penalty(&self) -> usize {
        self.evidence_rank_penalty
    }
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            top_k: 5,
            judge: RerankJudge::None,
            evidence_rank_penalty: 10,
        }
    }
}

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct RawRerankConfig {
    top_k: usize,
    judge: serde_json::Value,
    evidence_rank_penalty: usize,
    candidates: Option<usize>,
}

impl Default for RawRerankConfig {
    fn default() -> Self {
        let config = RerankConfig::default();
        Self {
            top_k: config.top_k,
            judge: serde_json::to_value(config.judge)
                .expect("a judge is plain data, which always serializes"),
            evidence_rank_penalty: config.evidence_rank_penalty,
            candidates: None,
        }
    }
}

impl TryFrom<RawRerankConfig> for RerankConfig {
    type Error = InvalidRerankConfig;

    fn try_from(raw: RawRerankConfig) -> Result<Self, Self::Error> {
        if let Some(candidates) = raw.candidates {
            return Err(InvalidRerankConfig::CandidatesWithoutJudge(candidates));
        }
        if raw.judge == "system_one" {
            return Err(InvalidRerankConfig::SystemOneWithoutCandidates);
        }
        let judge = serde_json::from_value(raw.judge).map_err(InvalidRerankConfig::Judge)?;
        Self::new(raw.top_k, judge, raw.evidence_rank_penalty)
    }
}

/// Threshold overrides per category; a step a category does not list uses [`PipelineConfig::thresholds`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PerCategory {
    /// For [`CategoryKind::Boolean`] runs.
    #[serde(skip_serializing_if = "Thresholds::is_empty")]
    pub boolean: Thresholds,
    /// For [`CategoryKind::Choice`] runs.
    #[serde(skip_serializing_if = "Thresholds::is_empty")]
    pub choice: Thresholds,
    /// For [`CategoryKind::Ranking`] runs.
    #[serde(skip_serializing_if = "Thresholds::is_empty")]
    pub ranking: Thresholds,
    /// For [`CategoryKind::Open`] runs.
    #[serde(skip_serializing_if = "Thresholds::is_empty")]
    pub open: Thresholds,
}

impl PerCategory {
    fn get(&self, category: CategoryKind) -> &Thresholds {
        match category {
            CategoryKind::Boolean => &self.boolean,
            CategoryKind::Choice => &self.choice,
            CategoryKind::Ranking => &self.ranking,
            CategoryKind::Open => &self.open,
        }
    }

    /// The category's override, else `general`. A run with no category uses `general`.
    pub fn threshold(
        &self,
        general: &Thresholds,
        category: Option<CategoryKind>,
        step: StepName,
    ) -> Option<Probability> {
        category
            .and_then(|category| self.get(category).get(step))
            .or_else(|| general.get(step))
    }
}

impl PipelineConfig {
    /// Reads and parses the file at `path`.
    ///
    /// # Errors
    ///
    /// When the file cannot be read, or as [`PipelineConfig::parse`].
    pub fn read(path: &Path) -> eros::Result<Self> {
        read(path)
    }

    /// Parses JSON and checks invariants across fields.
    ///
    /// # Errors
    ///
    /// On invalid JSON, an unknown key, a probability outside `0..=1`, an unthresholded step name, an
    /// [`InvalidRerankConfig`], or an [`InvalidMaxTokens`].
    pub fn parse(text: &str) -> eros::Result<Self> {
        Ok(serde_json::from_str(text)?)
    }

    /// The category's override, else the general threshold.
    pub fn threshold(&self, category: CategoryKind, step: StepName) -> Option<Probability> {
        self.per_category
            .threshold(&self.thresholds, Some(category), step)
    }
}

// `#[eros::context]` on an associated function without `self` calls its inner function unqualified.
#[eros::context("reading pipeline config {}", path.display())]
fn read(path: &Path) -> eros::Result<PipelineConfig> {
    let text = fs::read_to_string(path)?;
    PipelineConfig::parse(&text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::hybrid::Bm25;

    #[test]
    fn shipped_config_parses() {
        let config = PipelineConfig::parse(include_str!("../config/drew.json")).unwrap();
        assert!(config.classify.min_confidence.is_some());
        assert!(
            config
                .threshold(CategoryKind::Boolean, StepName::Answer)
                .is_some()
        );
    }

    #[test]
    fn an_empty_path_is_unset_and_a_missing_one_is_an_error() {
        let empty = ConfigArgs {
            path: Some(PathBuf::new()),
        };
        assert!(empty.load().is_ok());
        let missing = ConfigArgs {
            path: Some("no/such/drew.json".into()),
        };
        assert!(missing.load().is_err());
    }

    #[test]
    fn empty_object_is_the_defaults() {
        assert_eq!(
            PipelineConfig::parse("{}").unwrap(),
            PipelineConfig::default()
        );
    }

    #[test]
    fn category_override_wins_over_the_general_threshold() {
        let config = PipelineConfig::parse(
            r#"{"thresholds": {"answer": 0.6}, "per_category": {"boolean": {"answer": 0.8}}}"#,
        )
        .unwrap();
        assert_eq!(
            config.threshold(CategoryKind::Boolean, StepName::Answer),
            Some(Probability::new(0.8).unwrap())
        );
        assert_eq!(
            config.threshold(CategoryKind::Open, StepName::Answer),
            Some(Probability::new(0.6).unwrap())
        );
        assert_eq!(config.threshold(CategoryKind::Open, StepName::Rerank), None);
    }

    #[test]
    fn typos_and_unthresholded_steps_are_rejected() {
        for bad in [
            r#"{"thresold": {}}"#,
            r#"{"rerank": {"top": 5}}"#,
            r#"{"thresholds": {"classify": 0.5}}"#,
            r#"{"per_category": {"boolean": {"explain": 0.5}}}"#,
        ] {
            assert!(PipelineConfig::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_rerank_judge_is_none_or_system_one_with_its_candidates() {
        let config = PipelineConfig::parse(include_str!("../config/drew.json")).unwrap();
        assert_eq!(config.rerank.judge(), RerankJudge::None);
        assert_eq!(RerankConfig::default().judge(), RerankJudge::None);
        let config = PipelineConfig::parse(
            r#"{"rerank": {"top_k": 5, "judge": {"system_one": {"candidates": 25}}}}"#,
        )
        .unwrap();
        assert_eq!(
            config.rerank.judge(),
            RerankJudge::SystemOne(SystemOneJudge { candidates: 25 })
        );
        assert_eq!(
            PipelineConfig::parse(r#"{"rerank": {"judge": "none", "top_k": 50}}"#)
                .unwrap()
                .rerank
                .top_k(),
            50
        );
    }

    #[test]
    fn invalid_rerank_judges_are_rejected_on_every_load_path() {
        for (bad, says) in [
            (
                r#"{"rerank": {"candidates": 25}}"#,
                r#""judge": {"system_one": {"candidates": 25}}"#,
            ),
            (
                r#"{"rerank": {"judge": "none", "candidates": 25}}"#,
                "applies only to the System One judge",
            ),
            (
                r#"{"rerank": {"top_k": 10, "judge": {"system_one": {"candidates": 5}}}}"#,
                "rerank.top_k (10) exceeds the 5 candidates",
            ),
            (
                r#"{"rerank": {"top_k": 0}}"#,
                "rerank.top_k must be at least 1",
            ),
            (r#"{"rerank": {"judge": "kev"}}"#, "unknown variant `kev`"),
            (
                r#"{"rerank": {"judge": "system_one"}}"#,
                r#"needs its candidates; write "judge": {"system_one": {"candidates": 25}}"#,
            ),
            (
                r#"{"rerank": {"judge": {"system_one": {}}}}"#,
                "missing field `candidates`",
            ),
            (
                r#"{"rerank": {"judge": {"system_one": {"candidates": 25, "top_k": 5}}}}"#,
                "unknown field `top_k`",
            ),
        ] {
            let error = format!("{:#}", PipelineConfig::parse(bad).unwrap_err());
            assert!(error.contains(says), "{bad}: {error}");
            assert!(
                serde_json::from_str::<PipelineConfig>(bad).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn a_misspelled_step_is_rejected_by_name() {
        for (bad, key) in [
            (r#"{"thresholds": {"anser": 0.7}}"#, "anser"),
            (
                r#"{"per_category": {"boolean": {"rerankk": 0.7}}}"#,
                "rerankk",
            ),
        ] {
            let error = format!("{:#}", PipelineConfig::parse(bad).unwrap_err());
            assert!(error.contains(&format!("`{key}`")), "{error}");
        }
    }

    #[test]
    fn step_names_match_the_steps_they_threshold() {
        use crate::pipeline::{
            Boolean, Open, Step,
            steps::{Answer, Rerank},
        };
        assert_eq!(StepName::Rerank.as_str(), <Rerank<Open> as Step>::NAME);
        assert_eq!(StepName::Answer.as_str(), <Answer<Boolean> as Step>::NAME);
    }

    #[test]
    fn shipped_config_round_trips() {
        let config = PipelineConfig::parse(include_str!("../config/drew.json")).unwrap();
        let text = serde_json::to_string(&config).unwrap();
        assert_eq!(PipelineConfig::parse(&text).unwrap(), config);
    }

    #[test]
    fn probabilities_outside_zero_to_one_are_rejected_on_every_load_path() {
        for bad in [
            r#"{"thresholds": {"answer": 1.5}}"#,
            r#"{"thresholds": {"rerank": -0.1}}"#,
            r#"{"thresholds": {"answer": 1e39}}"#,
            r#"{"per_category": {"boolean": {"answer": 1.01}}}"#,
            r#"{"classify": {"min_confidence": 1.2}}"#,
            r#"{"classify": {"min_confidence": -0.5}}"#,
        ] {
            assert!(PipelineConfig::parse(bad).is_err(), "{bad}");
            assert!(
                serde_json::from_str::<PipelineConfig>(bad).is_err(),
                "{bad}"
            );
        }
        assert!(PipelineConfig::parse(r#"{"thresholds": {"answer": 1.0}}"#).is_ok());
    }

    #[test]
    fn the_hosted_config_differs_from_the_shipped_one_only_in_explain_max_tokens() {
        let shipped = PipelineConfig::parse(include_str!("../config/drew.json")).unwrap();
        let mut hosted =
            PipelineConfig::parse(include_str!("../config/drew-workers-ai.json")).unwrap();
        hosted.explain.max_tokens = shipped.explain.max_tokens;
        assert_eq!(
            hosted, shipped,
            "config/drew-workers-ai.json and config/drew.json differ in more than explain.max_tokens; change both \
             files, or record the deliberate difference in this test"
        );
    }

    #[test]
    fn explain_max_tokens_defaults_to_2048_and_is_bounded_on_every_load_path() {
        assert_eq!(PipelineConfig::default().explain.max_tokens.get(), 2048);
        let shipped = PipelineConfig::parse(include_str!("../config/drew.json")).unwrap();
        assert_eq!(shipped.explain.max_tokens, MaxTokens::DEFAULT);
        let hosted = PipelineConfig::parse(include_str!("../config/drew-workers-ai.json")).unwrap();
        assert_eq!(hosted.explain.max_tokens.get(), 8192);
        let max = PipelineConfig::parse(r#"{"explain": {"max_tokens": 65536}}"#).unwrap();
        assert_eq!(max.explain.max_tokens.get(), MaxTokens::MAX);

        for (bad, says) in [
            (
                r#"{"explain": {"max_tokens": 0}}"#,
                "explain.max_tokens must be between 1 and 65536, got 0",
            ),
            (
                r#"{"explain": {"max_tokens": 65537}}"#,
                "explain.max_tokens must be between 1 and 65536, got 65537",
            ),
            (r#"{"explain": {"max_tokens": -1}}"#, "invalid value"),
            (r#"{"explain": {"max_tokens": 2048.5}}"#, "invalid type"),
            (r#"{"explain": {"max_token": 2048}}"#, "unknown field"),
        ] {
            let error = format!("{:#}", PipelineConfig::parse(bad).unwrap_err());
            assert!(error.contains(says), "{bad}: {error}");
            assert!(
                serde_json::from_str::<PipelineConfig>(bad).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn bm25_constants_are_checked_on_every_load_path() {
        for bad in [
            r#"{"retrieval": {"bm25": {"k1": -0.1}}}"#,
            r#"{"retrieval": {"bm25": {"b": 1.5}}}"#,
            r#"{"retrieval": {"bm25": {"b": -0.5}}}"#,
            r#"{"retrieval": {"bm25": {"k1": 1e309}}}"#,
            r#"{"retrieval": {"bm25": {"k": 1.2}}}"#,
        ] {
            assert!(PipelineConfig::parse(bad).is_err(), "{bad}");
            assert!(
                serde_json::from_str::<PipelineConfig>(bad).is_err(),
                "{bad}"
            );
        }
        let config = PipelineConfig::parse(r#"{"retrieval": {"bm25": {"b": 0.0}}}"#).unwrap();
        assert_eq!(config.retrieval.bm25, Bm25::new(1.2, 0.0).unwrap());
    }
}
