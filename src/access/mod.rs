//! Access control: who is asking ([`Principal`]), what the policy lets them read ([`Access`]), and the two ports
//! that decide it, [`Authorizer`] and [`Labeler`]. See `docs/design.md` section 8.
//!
//! The policy is compiled once per run into [`Access`], a data predicate over chunk [`Labels`]. The same
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

/// Which tenants' chunks a grant admits. A chunk without a tenant passes every grant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TenantScope {
    /// Every tenant's chunks.
    Any,
    /// Only these tenants' chunks; empty admits untenanted chunks only.
    Only(BTreeSet<String>),
}

impl Default for TenantScope {
    fn default() -> Self {
        Self::Only(BTreeSet::new())
    }
}

impl TenantScope {
    fn admits(&self, tenant: Option<&str>) -> bool {
        match (tenant, self) {
            (None, _) | (Some(_), Self::Any) => true,
            (Some(tenant), Self::Only(tenants)) => tenants.contains(tenant),
        }
    }
}

/// `"any"` or a list of tenant names.
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum TenantScopeRepr {
    Any(AnyTenant),
    Only(BTreeSet<String>),
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum AnyTenant {
    Any,
}

impl Serialize for TenantScope {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Any => TenantScopeRepr::Any(AnyTenant::Any),
            Self::Only(tenants) => TenantScopeRepr::Only(tenants.clone()),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for TenantScope {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(match TenantScopeRepr::deserialize(deserializer)? {
            TenantScopeRepr::Any(AnyTenant::Any) => Self::Any,
            TenantScopeRepr::Only(tenants) => Self::Only(tenants),
        })
    }
}

/// One role's permission to read chunks: up to `clearance`, of the tenants in `tenants`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkGrant {
    role: Role,
    clearance: Sensitivity,
    tenants: TenantScope,
}

impl ChunkGrant {
    /// `role` may read chunks of sensitivity up to `clearance` whose tenant `tenants` admits.
    pub fn new(role: Role, clearance: Sensitivity, tenants: TenantScope) -> Self {
        Self {
            role,
            clearance,
            tenants,
        }
    }

    /// The role the grant comes from; named by every decision it allows.
    pub fn role(&self) -> &Role {
        &self.role
    }

    /// The most sensitive tier it admits.
    pub fn clearance(&self) -> Sensitivity {
        self.clearance
    }

    /// The tenants it admits.
    pub fn tenants(&self) -> &TenantScope {
        &self.tenants
    }

    /// Every sensitivity up to the clearance, as the store compares them.
    pub fn sensitivities(&self) -> impl Iterator<Item = Sensitivity> + '_ {
        Sensitivity::ALL
            .into_iter()
            .filter(|sensitivity| *sensitivity <= self.clearance)
    }

    fn admits(&self, labels: &Labels) -> bool {
        labels.sensitivity <= self.clearance && self.tenants.admits(labels.tenant.as_deref())
    }
}

/// The chunks one principal may read, as a predicate over [`Labels`]. A chunk is readable when its ACL is empty or
/// names one of `members`, and some grant admits its sensitivity and tenant. No grant: nothing is readable.
///
/// Retrievers push it into their queries, and every step rechecks with [`ChunkScope::decide`]; both read the same
/// fields, so they agree.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkScope {
    members: BTreeSet<String>,
    grants: Vec<ChunkGrant>,
}

impl ChunkScope {
    /// Admits no chunk.
    pub fn none() -> Self {
        Self::default()
    }

    /// `members` are the ACL entries that name the principal: its id and its roles. A chunk is allowed by the first
    /// of `grants`, in order, that admits it.
    pub fn new(
        members: impl IntoIterator<Item = impl Into<String>>,
        grants: impl IntoIterator<Item = ChunkGrant>,
    ) -> Self {
        Self {
            members: members.into_iter().map(Into::into).collect(),
            grants: grants.into_iter().collect(),
        }
    }

    /// The ACL entries that name the principal, sorted.
    pub fn members(&self) -> &BTreeSet<String> {
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
        if !labels.acl.is_empty() && !labels.acl.iter().any(|entry| self.members.contains(entry)) {
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
    /// The chunk's ACL names neither the principal nor any of its roles.
    NotInAcl,
    /// No grant admits it: the principal has no role for its sensitivity, tenant, or for runs at all.
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
