//! Access control: who is asking ([`Principal`]), what the policy lets them read ([`Access`]), and the two ports
//! that decide it, [`Authorizer`] and [`Labeler`]. See `docs/design.md` section 8.
//!
//! The policy is compiled once per run into [`Access`], a data predicate over chunk [`Labels`]. Its vocabulary is
//! typed: an [`AclEntry`] is a user or a role, never text that could be either. The same
//! [`ChunkScope`] is pushed into the retriever's queries and rechecked in memory by every step that touches chunks,
//! so a pushdown and a recheck cannot disagree. Deny by default: a principal the policy does not know gets
//! [`Access::none`], and a policy that cannot answer fails the run before anything is retrieved.
//!
//! [`rules::RulePolicy`] and [`rules::LabelRules`] are the shipped adapters, configured by `config/access.json`; an
//! outside adapter (an IdP, a policy engine) implements the ports and runs their contract suites,
//! `access::contract`, `recall::authorizer_contract!` and `recall::labeler_contract!` (`testing` feature).

#[cfg(any(test, feature = "testing"))]
pub mod contract;
/// In-memory and failing [`Authorizer`]s and [`Labeler`]s for tests.
#[cfg(any(test, feature = "testing"))]
pub mod fakes;
pub mod rules;

use std::{collections::BTreeSet, fmt};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::domain::{Chunk, ChunkRef, Document, Labels, Sensitivity, UserId};

/// A role name: one or more of `a-z`, `0-9`, `_`, `-`, `.` and `:`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Role(String);

/// Text rejected by [`Role::new`].
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[error("invalid role {0:?}: expected one or more of a-z, 0-9, `_`, `-`, `.` and `:`")]
pub struct InvalidRole(pub String);

impl Role {
    /// # Errors
    ///
    /// [`InvalidRole`] when `name` is empty or has a character outside the allowed set.
    pub fn new(name: impl Into<String>) -> Result<Self, InvalidRole> {
        let name = name.into();
        let valid = !name.is_empty()
            && name
                .bytes()
                .all(|b| matches!(b, b'a'..=b'z' | b'0'..=b'9' | b'_' | b'-' | b'.' | b':'));
        if valid {
            Ok(Self(name))
        } else {
            Err(InvalidRole(name))
        }
    }

    /// The name, as written in ACLs and config.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for Role {
    type Error = InvalidRole;

    fn try_from(name: String) -> Result<Self, InvalidRole> {
        Self::new(name)
    }
}

impl From<Role> for String {
    fn from(role: Role) -> String {
        role.0
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// One entry of a chunk's ACL: a user or a role, never text that could be either. Written `user:<id>` or
/// `role:<name>` in config, in `chunks.acl` and in traces, so a user whose id is `hr` is not a member of role `hr`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum AclEntry {
    /// Exactly this user.
    User(UserId),
    /// Every principal holding this role.
    Role(Role),
}

/// Text rejected by [`AclEntry`]'s `FromStr`.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[error("invalid ACL entry {0:?}: expected `user:<id>` or `role:<name>`")]
pub struct InvalidAclEntry(pub String);

impl std::str::FromStr for AclEntry {
    type Err = InvalidAclEntry;

    fn from_str(entry: &str) -> Result<Self, InvalidAclEntry> {
        let invalid = || InvalidAclEntry(entry.to_owned());
        match entry.split_once(':') {
            Some(("user", id)) => UserId::new(id).map(Self::User).map_err(|_| invalid()),
            Some(("role", name)) => Role::new(name).map(Self::Role).map_err(|_| invalid()),
            _ => Err(invalid()),
        }
    }
}

impl TryFrom<String> for AclEntry {
    type Error = InvalidAclEntry;

    fn try_from(entry: String) -> Result<Self, InvalidAclEntry> {
        entry.parse()
    }
}

impl From<AclEntry> for String {
    fn from(entry: AclEntry) -> String {
        entry.to_string()
    }
}

impl fmt::Display for AclEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::User(id) => write!(f, "user:{id}"),
            Self::Role(role) => write!(f, "role:{role}"),
        }
    }
}

/// Who may read a chunk, as its source system said: readable by a principal that is one of the entries, or by
/// anyone when there are none. Grants apply on top either way.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Acl(BTreeSet<AclEntry>);

/// Two restricting ACLs with no entry in common, which [`Acl::tighten`] cannot narrow to one.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[error("the ACLs {0} and {1} share no entry, so no principal could read under both")]
pub struct DisjointAcls(pub Acl, pub Acl);

impl Acl {
    /// No ACL restriction.
    pub fn anyone() -> Self {
        Self::default()
    }

    /// Readable only by these entries; none is [`Acl::anyone`].
    pub fn only(entries: impl IntoIterator<Item = AclEntry>) -> Self {
        Self(entries.into_iter().collect())
    }

    /// Sorted; empty when anyone may read.
    pub fn entries(&self) -> &BTreeSet<AclEntry> {
        &self.0
    }

    /// Whether the ACL lets a principal named by `members` read.
    pub fn admits(&self, members: &BTreeSet<AclEntry>) -> bool {
        self.0.is_empty() || !self.0.is_disjoint(members)
    }

    /// An ACL no wider than either: the other one when one is open, else the entries in both.
    ///
    /// # Errors
    ///
    /// [`DisjointAcls`] when both restrict and share no entry; an empty result would read as open.
    pub fn tighten(&self, other: &Acl) -> Result<Acl, DisjointAcls> {
        if self.0.is_empty() {
            return Ok(other.clone());
        }
        if other.0.is_empty() {
            return Ok(self.clone());
        }
        let both: BTreeSet<AclEntry> = self.0.intersection(&other.0).cloned().collect();
        if both.is_empty() {
            Err(DisjointAcls(self.clone(), other.clone()))
        } else {
            Ok(Self(both))
        }
    }

    /// Each entry as stored in `chunks.acl`.
    pub fn to_strings(&self) -> Vec<String> {
        self.0.iter().map(ToString::to_string).collect()
    }
}

impl TryFrom<Vec<String>> for Acl {
    type Error = InvalidAclEntry;

    fn try_from(entries: Vec<String>) -> Result<Self, InvalidAclEntry> {
        entries
            .into_iter()
            .map(AclEntry::try_from)
            .collect::<Result<_, _>>()
            .map(Self)
    }
}

impl fmt::Display for Acl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        for (i, entry) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{entry}")?;
        }
        f.write_str("]")
    }
}

/// The authenticated caller: an id, and the roles its authenticator asserts. Built once at the boundary (the HTTP
/// API's bearer key, `recall eval --as`); the [`Authorizer`] decides what it may read.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Principal {
    id: UserId,
    roles: BTreeSet<Role>,
}

impl Principal {
    /// A principal with no asserted roles.
    pub fn new(id: UserId) -> Self {
        Self {
            id,
            roles: BTreeSet::new(),
        }
    }

    /// Adds roles the authenticator vouches for, such as an identity provider's group claims.
    pub fn with_roles(mut self, roles: impl IntoIterator<Item = Role>) -> Self {
        self.roles.extend(roles);
        self
    }

    /// Who is asking; recorded as the run's user.
    pub fn id(&self) -> &UserId {
        &self.id
    }

    /// The asserted roles, sorted.
    pub fn roles(&self) -> &BTreeSet<Role> {
        &self.roles
    }
}

/// One role's permission to read chunks: up to `clearance`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkGrant {
    role: Role,
    clearance: Sensitivity,
}

impl ChunkGrant {
    /// `role` may read chunks of sensitivity up to `clearance`.
    pub fn new(role: Role, clearance: Sensitivity) -> Self {
        Self { role, clearance }
    }

    /// The role the grant comes from; named by every decision it allows.
    pub fn role(&self) -> &Role {
        &self.role
    }

    /// The most sensitive tier it admits.
    pub fn clearance(&self) -> Sensitivity {
        self.clearance
    }

    /// Every sensitivity up to the clearance, as the store compares them.
    pub fn sensitivities(&self) -> impl Iterator<Item = Sensitivity> + '_ {
        Sensitivity::ALL
            .into_iter()
            .filter(|sensitivity| *sensitivity <= self.clearance)
    }

    fn admits(&self, labels: &Labels) -> bool {
        labels.sensitivity <= self.clearance
    }
}

/// The chunks one principal may read, as a predicate over [`Labels`]. A chunk is readable when its ACL admits one of
/// `members`, and some grant clears its sensitivity. No grant: nothing is readable.
///
/// The members name one user, the principal, and its roles; never a second user. Retrievers push the scope into
/// their queries, and every step rechecks with [`ChunkScope::decide`]; both compare the same tagged entries, so
/// they agree.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "RawChunkScope")]
pub struct ChunkScope {
    members: BTreeSet<AclEntry>,
    grants: Vec<ChunkGrant>,
}

#[derive(Deserialize)]
struct RawChunkScope {
    members: BTreeSet<AclEntry>,
    grants: Vec<ChunkGrant>,
}

/// A stored [`ChunkScope`] whose members name more than one user.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[error("a chunk scope names {0} users; it names at most its principal")]
pub struct TooManyUsers(pub usize);

impl TryFrom<RawChunkScope> for ChunkScope {
    type Error = TooManyUsers;

    fn try_from(raw: RawChunkScope) -> Result<Self, TooManyUsers> {
        let users = raw
            .members
            .iter()
            .filter(|entry| matches!(entry, AclEntry::User(_)))
            .count();
        if users > 1 {
            return Err(TooManyUsers(users));
        }
        Ok(Self {
            members: raw.members,
            grants: raw.grants,
        })
    }
}

impl ChunkScope {
    /// Admits no chunk.
    pub fn none() -> Self {
        Self::default()
    }

    /// The scope of the principal `user` holding `roles`: an ACL admits it when it names `user` or one of `roles`.
    /// A chunk is allowed by the first of `grants`, in order, that admits it.
    pub fn new(
        user: UserId,
        roles: impl IntoIterator<Item = Role>,
        grants: impl IntoIterator<Item = ChunkGrant>,
    ) -> Self {
        Self {
            members: std::iter::once(AclEntry::User(user))
                .chain(roles.into_iter().map(AclEntry::Role))
                .collect(),
            grants: grants.into_iter().collect(),
        }
    }

    /// The ACL entries that name the principal, sorted: at most one user, then its roles.
    pub fn members(&self) -> &BTreeSet<AclEntry> {
        &self.members
    }

    /// In the order they are tried.
    pub fn grants(&self) -> &[ChunkGrant] {
        &self.grants
    }

    /// Whether a chunk with these labels is readable.
    pub fn admits(&self, labels: &Labels) -> bool {
        self.decide(labels).is_allowed()
    }

    /// Allowed by the first grant that admits the labels, or denied with the first rule that failed: the ACL, then
    /// the grants.
    pub fn decide(&self, labels: &Labels) -> Decision {
        if !labels.acl.admits(&self.members) {
            return Decision::Denied {
                reason: DenyReason::NotInAcl,
            };
        }
        match self.grants.iter().find(|grant| grant.admits(labels)) {
            Some(grant) => Decision::Allowed {
                role: grant.role.clone(),
            },
            None => Decision::Denied {
                reason: DenyReason::NoGrant,
            },
        }
    }

    /// One decision per chunk, in order, for the trace.
    pub fn check<'a>(&self, chunks: impl IntoIterator<Item = &'a Chunk>) -> Vec<AccessDecision> {
        chunks
            .into_iter()
            .map(|chunk| AccessDecision {
                chunk: chunk.reference(),
                decision: self.decide(&chunk.labels),
            })
            .collect()
    }
}

/// Whose persisted runs, with their traces, a principal may read.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "runs", rename_all = "snake_case")]
pub enum RunAccess {
    /// None.
    #[default]
    Nothing,
    /// Its own.
    Own {
        /// The role that grants it.
        role: Role,
    },
    /// Everyone's, for auditing.
    Any {
        /// The role that grants it.
        role: Role,
    },
}

impl RunAccess {
    /// Whether `principal` may read a run `owner` asked.
    pub fn decide(&self, principal: &Principal, owner: &UserId) -> Decision {
        match self {
            Self::Any { role } => Decision::Allowed { role: role.clone() },
            Self::Own { role } if principal.id() == owner => {
                Decision::Allowed { role: role.clone() }
            }
            Self::Own { .. } => Decision::Denied {
                reason: DenyReason::NotOwner,
            },
            Self::Nothing => Decision::Denied {
                reason: DenyReason::NoGrant,
            },
        }
    }
}

/// What one principal may do, compiled by an [`Authorizer`] once per run and recorded in its trace.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Access {
    chunks: ChunkScope,
    runs: RunAccess,
}

impl Access {
    /// Nothing: no chunk and no run. What the policy gives a principal it does not know.
    pub fn none() -> Self {
        Self::default()
    }

    /// See the field getters.
    pub fn new(chunks: ChunkScope, runs: RunAccess) -> Self {
        Self { chunks, runs }
    }

    /// The chunks it may read.
    pub fn chunks(&self) -> &ChunkScope {
        &self.chunks
    }

    /// The runs it may read.
    pub fn runs(&self) -> &RunAccess {
        &self.runs
    }
}

/// The outcome of one access check.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "effect", rename_all = "snake_case")]
pub enum Decision {
    /// Readable.
    Allowed {
        /// The role whose grant allowed it.
        role: Role,
    },
    /// Not readable.
    Denied {
        /// The first rule that failed.
        reason: DenyReason,
    },
}

impl Decision {
    /// Whether the check allowed it.
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed { .. })
    }
}

/// Why a check denied.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DenyReason {
    /// The chunk's ACL names neither the principal's user nor any of its roles.
    NotInAcl,
    /// No grant admits it: the principal has no role for its sensitivity, or for runs at all.
    NoGrant,
    /// The run belongs to another principal, and the grant covers its own runs only.
    NotOwner,
}

/// One chunk's check at one step, recorded in that step's trace entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccessDecision {
    /// The exact version checked.
    pub chunk: ChunkRef,
    /// Allowed or denied, and why.
    pub decision: Decision,
}

/// Which policy decided, recorded in every run's `authorize` entry. Two refs are the same policy only if both fields
/// match.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PolicyRef {
    /// The adapter, e.g. `rules`.
    pub id: String,
    /// Changes whenever the policy's decisions can change, e.g. a hash of its config.
    pub version: String,
}

/// Decides what a principal may read. Injected at assembly; every run asks it once, before retrieval.
#[async_trait]
pub trait Authorizer: Send + Sync {
    /// Recorded in the trace of every run.
    fn policy(&self) -> PolicyRef;

    /// What `principal` may read. Deny by default: a principal the policy does not know gets [`Access::none`].
    ///
    /// # Errors
    ///
    /// When the policy cannot answer, e.g. its backend is down. The run then fails before retrieval.
    async fn access(&self, principal: &Principal) -> eros::Result<Access>;
}

/// Sets a document's [`Labels`] at ingestion, before its chunks are versioned and stored. Labels are part of each
/// chunk's version, so a labeler change re-versions the chunks it relabels.
#[async_trait]
pub trait Labeler: Send + Sync {
    /// The labels every chunk of `document` is stored with. Depends only on `document`.
    ///
    /// # Errors
    ///
    /// When the labels cannot be decided; the document is then not stored, and its source's ingestion fails.
    async fn label(&self, document: &Document) -> eros::Result<Labels>;
}

/// Keeps the labels the connector set, such as an ACL inherited from the source system.
#[derive(Clone, Copy, Debug, Default)]
pub struct ConnectorLabels;

#[async_trait]
impl Labeler for ConnectorLabels {
    async fn label(&self, document: &Document) -> eros::Result<Labels> {
        Ok(document.labels.clone())
    }
}

#[cfg(test)]
mod tests;
