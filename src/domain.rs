//! Types shared by every module: questions, chunks, labels, provenance, and probabilities.

use std::fmt;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::llm::ModelRef;

/// The user's question, verbatim. [`Span`]s in [`Provenance::UserPrompt`] index into its bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Question(String);

impl Question {
    /// Wraps the text as given; no trimming or validation.
    pub fn new(text: impl Into<String>) -> Self {
        Self(text.into())
    }

    /// The question text.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The text a `Span` points at, if the span is in bounds and on char boundaries.
    pub fn slice(&self, span: Span) -> Option<&str> {
        self.0.get(span.start..span.end)
    }
}

/// Identity of one pipeline run, persisted as `pipeline_runs.id` and shown to users on failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RunId(Uuid);

impl RunId {
    /// A fresh UUIDv7, so ids sort by creation time.
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }

    /// The underlying UUID, for binding in queries.
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for RunId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Uuid> for RunId {
    fn from(id: Uuid) -> Self {
        Self(id)
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Who asked: the user mapped from an API bearer key, or `slack:{user_id}` for Slack runs.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct UserId(pub String);

/// The runtime value of a question category; [`crate::pipeline`] has one marker type per variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CategoryKind {
    /// A yes/no question.
    Boolean,
    /// Pick one of at least two candidates.
    Choice,
    /// Order items by a criterion stated in the question.
    Ranking,
    /// Anything else: answered by the LLM alone, with no System One verdict.
    Open,
}

impl CategoryKind {
    /// Every variant, in declaration order.
    pub const ALL: [Self; 4] = [Self::Boolean, Self::Choice, Self::Ranking, Self::Open];

    /// The serde name, also stored in `pipeline_runs.category`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Boolean => "boolean",
            Self::Choice => "choice",
            Self::Ranking => "ranking",
            Self::Open => "open",
        }
    }
}

impl TryFrom<&str> for CategoryKind {
    type Error = UnknownVariant;

    fn try_from(name: &str) -> Result<Self, UnknownVariant> {
        match name {
            "boolean" => Ok(Self::Boolean),
            "choice" => Ok(Self::Choice),
            "ranking" => Ok(Self::Ranking),
            "open" => Ok(Self::Open),
            other => Err(UnknownVariant::new("category", other)),
        }
    }
}

/// A string that names no variant of a text-backed domain enum.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[error("unknown {kind} {name:?}")]
pub struct UnknownVariant {
    /// Which enum was parsed, e.g. `"category"`.
    pub kind: &'static str,
    /// The rejected text.
    pub name: String,
}

impl UnknownVariant {
    fn new(kind: &'static str, name: &str) -> Self {
        Self {
            kind,
            name: name.to_owned(),
        }
    }
}

/// A probability or a confidence: finite and in `0..=1`. Step scores and thresholds are compared as these, so a
/// model's out-of-range or `NaN` output cannot pass a threshold. Serialized as a bare number.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "f32", into = "f32")]
pub struct Probability(f32);

/// A value rejected by [`Probability::new`]: outside `0..=1` or `NaN`.
#[derive(Debug, PartialEq, thiserror::Error)]
#[error("expected a probability between 0 and 1, got {0}")]
pub struct NotAProbability(pub f32);

impl Probability {
    /// # Errors
    ///
    /// [`NotAProbability`] when `value` is `NaN`, infinite, or outside `0..=1`.
    pub fn new(value: f32) -> Result<Self, NotAProbability> {
        if (0.0..=1.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err(NotAProbability(value))
        }
    }

    /// The value, in `0..=1`.
    pub fn get(self) -> f32 {
        self.0
    }

    /// `1 - p`: the probability of the other answer to a yes/no question.
    pub fn complement(self) -> Self {
        Self(1.0 - self.0)
    }
}

impl TryFrom<f32> for Probability {
    type Error = NotAProbability;

    fn try_from(value: f32) -> Result<Self, NotAProbability> {
        Self::new(value)
    }
}

impl From<Probability> for f32 {
    fn from(probability: Probability) -> f32 {
        probability.0
    }
}

/// Where a document came from. The markdown corpus is the only source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    /// A file in the pre-audited markdown directory.
    Markdown,
}

impl SourceKind {
    /// Every variant, in declaration order.
    pub const ALL: [Self; 1] = [Self::Markdown];

    /// The serde name, also stored in `documents.source` and used as the `DocumentId` prefix.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Markdown => "markdown",
        }
    }
}

impl TryFrom<&str> for SourceKind {
    type Error = UnknownVariant;

    fn try_from(name: &str) -> Result<Self, UnknownVariant> {
        match name {
            "markdown" => Ok(Self::Markdown),
            other => Err(UnknownVariant::new("source", other)),
        }
    }
}

/// A source unit: one markdown file in the docs corpus. Format: `{source}:{external_id}`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DocumentId(pub String);

impl DocumentId {
    /// `{source}:{external_id}`, with the source's [`SourceKind::as_str`] name.
    pub fn new(source: SourceKind, external_id: &str) -> Self {
        Self(format!("{}:{external_id}", source.as_str()))
    }
}

/// Stable identity of a chunk across re-ingestion. Format: `{document_id}#{locator}`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ChunkId(pub String);

impl ChunkId {
    /// `{document_id}#{locator}`; `locator` is [`ChunkDraft::locator`].
    pub fn new(document: &DocumentId, locator: &str) -> Self {
        Self(format!("{}#{locator}", document.0))
    }
}

/// Content address of one immutable chunk version: hex SHA-256 over its content, labels, and embedding model.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ChunkVersion(pub String);

impl ChunkVersion {
    /// Hashes everything that affects what retrieval returns for the chunk, so any change to the text, its labels,
    /// or the embedding model yields a new version.
    pub fn compute(document: &Document, draft: &ChunkDraft, embedding_model: &ModelRef) -> Self {
        #[derive(Serialize)]
        struct Content<'a> {
            title: &'a str,
            heading: &'a Option<String>,
            url: &'a str,
            text: &'a str,
            authoritative: bool,
            labels: &'a Labels,
            embedding_model: &'a ModelRef,
        }
        let content = Content {
            title: &document.title,
            heading: &draft.heading,
            url: &draft.url,
            text: &draft.text,
            authoritative: draft.authoritative,
            labels: &document.labels,
            embedding_model,
        };
        let bytes = serde_json::to_vec(&content)
            .expect("chunk content is plain data, which always serializes");
        Self(hex::encode(Sha256::digest(bytes)))
    }
}

/// One exact chunk version, as cited in traces so a run stays reproducible after re-ingestion.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkRef {
    /// Stable across versions.
    pub id: ChunkId,
    /// The content address of the version that was used.
    pub version: ChunkVersion,
}

/// How widely a chunk may be shown once the security phase enforces [`Labels`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sensitivity {
    /// No restriction.
    Public,
    /// Internal to the company; every markdown chunk is labelled this.
    Internal,
    /// Restricted to a Drew-owned maintainers group.
    Confidential,
}

impl Sensitivity {
    /// Every variant, in declaration order.
    pub const ALL: [Self; 3] = [Self::Public, Self::Internal, Self::Confidential];

    /// The serde name, also stored in `chunks.sensitivity`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Public => "public",
            Self::Internal => "internal",
            Self::Confidential => "confidential",
        }
    }
}

impl TryFrom<&str> for Sensitivity {
    type Error = UnknownVariant;

    fn try_from(name: &str) -> Result<Self, UnknownVariant> {
        match name {
            "public" => Ok(Self::Public),
            "internal" => Ok(Self::Internal),
            "confidential" => Ok(Self::Confidential),
            other => Err(UnknownVariant::new("sensitivity", other)),
        }
    }
}

/// Recorded on every chunk; enforced only in the security phase.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Labels {
    /// Principals inherited from the source ACL (groups, users, or channel membership).
    pub acl: Vec<String>,
    /// The customer the content concerns, if any; `None` means visible to every customer.
    pub customer: Option<String>,
    /// Visibility tier.
    pub sensitivity: Sensitivity,
}

/// What a source connector produces for one source unit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Document {
    /// The connector that produced it.
    pub source: SourceKind,
    /// Path relative to the docs root, with `/` separators.
    pub external_id: String,
    /// The first level-1 heading, else the file stem.
    pub title: String,
    /// Citation link for the whole file: `DREW_DOCS_BASE_URL` joined with the path, else the bare path.
    pub url: String,
    /// File modification time.
    pub updated_at: DateTime<Utc>,
    /// Copied onto every chunk.
    pub labels: Labels,
    /// In file order; locators are unique within the document.
    pub chunks: Vec<ChunkDraft>,
}

impl Document {
    /// See [`DocumentId::new`].
    pub fn id(&self) -> DocumentId {
        DocumentId::new(self.source, &self.external_id)
    }

    /// The id `draft` gets once stored; see [`ChunkId::new`].
    pub fn chunk_id(&self, draft: &ChunkDraft) -> ChunkId {
        ChunkId::new(&self.id(), &draft.locator)
    }
}

/// One heading section of a [`Document`], before it is versioned, embedded and stored as a [`Chunk`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChunkDraft {
    /// Unique within the document and stable across re-ingestion: the heading slug, or `intro` for text before
    /// the first heading.
    pub locator: String,
    /// Heading path joined with ` > `; `None` for the intro.
    pub heading: Option<String>,
    /// Deep link to the section.
    pub url: String,
    /// Section body, trimmed, without its heading line.
    pub text: String,
    /// Written by Drew.
    pub authoritative: bool,
}

/// A current chunk version as read back from the store.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    /// Stable across versions.
    pub id: ChunkId,
    /// Content address of this version.
    pub version: ChunkVersion,
    /// The document it was cut from.
    pub document_id: DocumentId,
    /// The document's source.
    pub source: SourceKind,
    /// The document's title.
    pub title: String,
    /// See [`ChunkDraft::heading`].
    pub heading: Option<String>,
    /// Deep link to the section.
    pub url: String,
    /// Section body; delimit it as data in prompts.
    pub text: String,
    /// Written by Drew; evidence chunks (`false`) rank below authoritative ones.
    pub authoritative: bool,
    /// The document's labels.
    pub labels: Labels,
}

impl Chunk {
    /// The id and version, for traces and citations.
    pub fn reference(&self) -> ChunkRef {
        ChunkRef {
            id: self.id.clone(),
            version: self.version.clone(),
        }
    }
}

/// Byte range into the question text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    /// Inclusive start offset.
    pub start: usize,
    /// Exclusive end offset.
    pub end: usize,
}

/// Where a cited value came from. System One never invents options, so every value has one of these.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provenance {
    /// Quoted from the question.
    UserPrompt(Span),
    /// Taken from a chunk retrieved in this run.
    Chunk(ChunkId),
}

/// A value with the sources it came from; a value without a source cannot be built.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cited<T> {
    /// The cited value.
    pub value: T,
    /// Where it came from.
    pub sources: AtLeastOne<Provenance>,
}

/// A non-empty list. Built with [`AtLeastOne::one`] or `TryFrom<Vec<T>>`, which fails with [`TooFew`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "Vec<T>", into = "Vec<T>")]
pub struct AtLeastOne<T: Clone>(Vec<T>);

/// A list of two or more, such as the candidates of a choice question. Built with `TryFrom<Vec<T>>`, which fails
/// with [`TooFew`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "Vec<T>", into = "Vec<T>")]
pub struct AtLeastTwo<T: Clone>(Vec<T>);

/// A list shorter than [`AtLeastOne`] or [`AtLeastTwo`] requires.
#[derive(Debug, thiserror::Error)]
#[error("expected at least {expected} items, got {actual}")]
pub struct TooFew {
    /// The minimum length.
    pub expected: usize,
    /// The length given.
    pub actual: usize,
}

macro_rules! at_least {
    ($name:ident, $n:literal) => {
        impl<T: Clone> $name<T> {
            /// The items, in order.
            pub fn as_slice(&self) -> &[T] {
                &self.0
            }

            /// The items, in order.
            pub fn into_vec(self) -> Vec<T> {
                self.0
            }
        }

        impl<T: Clone> TryFrom<Vec<T>> for $name<T> {
            type Error = TooFew;

            fn try_from(items: Vec<T>) -> Result<Self, TooFew> {
                if items.len() < $n {
                    return Err(TooFew {
                        expected: $n,
                        actual: items.len(),
                    });
                }
                Ok(Self(items))
            }
        }

        impl<T: Clone> From<$name<T>> for Vec<T> {
            fn from(items: $name<T>) -> Vec<T> {
                items.0
            }
        }
    };
}

at_least!(AtLeastOne, 1);
at_least!(AtLeastTwo, 2);

impl<T: Clone> AtLeastOne<T> {
    /// A list of exactly `item`.
    pub fn one(item: T) -> Self {
        Self(vec![item])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probability_rejects_out_of_range_and_non_finite_values() {
        for bad in [
            -0.3,
            1.7,
            -f32::EPSILON,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            assert!(Probability::new(bad).is_err(), "{bad}");
        }
        for good in [0.0, 0.5, 1.0] {
            assert_eq!(Probability::new(good).unwrap().get(), good);
        }
        assert_eq!(Probability::new(0.25).unwrap().complement().get(), 0.75);
    }

    #[test]
    fn probability_is_a_bare_number_and_deserialization_checks_it() {
        let p = Probability::new(0.875).unwrap();
        assert_eq!(serde_json::to_string(&p).unwrap(), "0.875");
        assert_eq!(serde_json::from_str::<Probability>("0.875").unwrap(), p);
        for bad in ["1.5", "-0.1", "1e39"] {
            assert!(serde_json::from_str::<Probability>(bad).is_err(), "{bad}");
        }
    }

    fn assert_text_matches_serde<T>(variants: &[T], as_str: fn(&T) -> &'static str)
    where
        T: Copy
            + fmt::Debug
            + PartialEq
            + Serialize
            + for<'a> TryFrom<&'a str, Error = UnknownVariant>,
    {
        for &variant in variants {
            let name = as_str(&variant);
            assert_eq!(serde_json::to_value(variant).unwrap(), name, "{variant:?}");
            assert_eq!(T::try_from(name).unwrap(), variant);
        }
        assert!(T::try_from("Unknown").is_err());
    }

    #[test]
    fn category_text_matches_serde_and_round_trips() {
        assert_text_matches_serde(&CategoryKind::ALL, CategoryKind::as_str);
    }

    #[test]
    fn sensitivity_text_matches_serde_and_round_trips() {
        assert_text_matches_serde(&Sensitivity::ALL, Sensitivity::as_str);
    }

    #[test]
    fn source_text_matches_serde_and_round_trips() {
        assert_text_matches_serde(&SourceKind::ALL, SourceKind::as_str);
    }

    #[test]
    fn at_least_two_rejects_one() {
        assert!(AtLeastTwo::try_from(vec![1]).is_err());
        assert!(AtLeastTwo::try_from(vec![1, 2]).is_ok());
    }

    #[test]
    fn at_least_one_deserialization_rejects_empty() {
        assert!(serde_json::from_str::<AtLeastOne<u8>>("[]").is_err());
        assert!(serde_json::from_str::<AtLeastOne<u8>>("[1]").is_ok());
    }

    #[test]
    fn chunk_version_changes_with_embedding_model() {
        let document = Document {
            source: SourceKind::Markdown,
            external_id: "roles.md".into(),
            title: "Roles".into(),
            url: "https://example.test/roles.md".into(),
            updated_at: Utc::now(),
            labels: Labels {
                acl: vec![],
                customer: None,
                sensitivity: Sensitivity::Internal,
            },
            chunks: vec![],
        };
        let draft = ChunkDraft {
            locator: "intro".into(),
            heading: None,
            url: document.url.clone(),
            text: "AGR_USERS maps users to roles.".into(),
            authoritative: true,
        };
        let titan = ModelRef::new("amazon.titan-embed-text-v2", "1");
        let cohere = ModelRef::new("cohere.embed-english-v3", "1");
        assert_ne!(
            ChunkVersion::compute(&document, &draft, &titan),
            ChunkVersion::compute(&document, &draft, &cohere),
        );
        assert_eq!(document.chunk_id(&draft).0, "markdown:roles.md#intro");
    }
}
