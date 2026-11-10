//! The shipped policy: roles and their grants, role membership and label rules, from `config/access.json`
//! (`RECALL_ACCESS`). [`RulePolicy`] is its [`Authorizer`], [`LabelRules`] its [`Labeler`].

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::domain::{Document, Labels, Sensitivity, UserId};

use super::{
    Access, Authorizer, ChunkGrant, ChunkScope, Labeler, PolicyRef, Principal, Role, RunAccess,
    TenantScope,
};

/// Where the shipped policy lives unless `RECALL_ACCESS` says otherwise.
pub const DEFAULT_PATH: &str = "config/access.json";

/// Where to load the [`AccessConfig`] from.
#[derive(Clone, Debug, clap::Args)]
pub struct AccessArgs {
    /// Access policy JSON: roles, members and label rules. Required: without it nobody may read anything.
    #[arg(id = "access", long = "access", env = "RECALL_ACCESS", default_value = DEFAULT_PATH)]
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

/// What one role may read.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoleGrant {
    /// The most sensitive tier it reads.
    pub clearance: Sensitivity,
    /// The tenants whose chunks it reads; untenanted chunks only when omitted.
    #[serde(default)]
    pub tenants: TenantScope,
    /// Whose persisted runs it reads; none when omitted.
    #[serde(default)]
    pub runs: RunGrant,
}

/// [`RoleGrant::runs`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunGrant {
    /// None.
    #[default]
    None,
    /// The principal's own.
    Own,
    /// Everyone's.
    Any,
}

/// Labels for every document whose id starts with `documents`, replacing the connector's.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LabelRule {
    /// A [`crate::domain::DocumentId`] prefix, e.g. `markdown:hr/`; the longest matching prefix wins.
    pub documents: String,
    /// See [`Labels::acl`].
    #[serde(default)]
    pub acl: Vec<String>,
    /// See [`Labels::tenant`].
    #[serde(default)]
    pub tenant: Option<String>,
    /// See [`Labels::sensitivity`].
    pub sensitivity: Sensitivity,
}

impl LabelRule {
    fn labels(&self) -> Labels {
        Labels {
            acl: self.acl.clone(),
            tenant: self.tenant.clone(),
            sensitivity: self.sensitivity,
        }
    }
}

/// The parsed policy. Every role named in `everyone` or `members` is defined in `roles`, and label prefixes are
/// non-empty and distinct; unknown keys are rejected.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "RawAccessConfig", into = "RawAccessConfig")]
pub struct AccessConfig {
    roles: BTreeMap<Role, RoleGrant>,
    everyone: BTreeSet<Role>,
    members: BTreeMap<UserId, BTreeSet<Role>>,
    labels: Vec<LabelRule>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAccessConfig {
    roles: BTreeMap<Role, RoleGrant>,
    #[serde(default)]
    everyone: BTreeSet<Role>,
    #[serde(default)]
    members: BTreeMap<UserId, BTreeSet<Role>>,
    #[serde(default)]
    labels: Vec<LabelRule>,
}

/// A policy whose parts disagree.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum InvalidAccessConfig {
    /// `everyone` or `members` names a role `roles` does not define.
    #[error("{used_by} names role {role}, which `roles` does not define")]
    UndefinedRole {
        /// The undefined role.
        role: Role,
        /// `everyone` or `members.<user>`.
        used_by: String,
    },
    /// A label rule with an empty `documents` prefix, which would relabel every document.
    #[error("a label rule has an empty `documents` prefix")]
    EmptyPrefix,
    /// Two label rules for one prefix.
    #[error("two label rules have the prefix {0:?}")]
    DuplicatePrefix(String),
}

impl TryFrom<RawAccessConfig> for AccessConfig {
    type Error = InvalidAccessConfig;

    fn try_from(raw: RawAccessConfig) -> Result<Self, InvalidAccessConfig> {
        let undefined = |roles: &BTreeSet<Role>, used_by: &dyn Fn() -> String| {
            roles
                .iter()
                .find(|role| !raw.roles.contains_key(*role))
                .map(|role| InvalidAccessConfig::UndefinedRole {
                    role: role.clone(),
                    used_by: used_by(),
                })
        };
        if let Some(error) = undefined(&raw.everyone, &|| "everyone".into()) {
            return Err(error);
        }
        for (user, roles) in &raw.members {
            if let Some(error) = undefined(roles, &|| format!("members.{}", user.0)) {
                return Err(error);
            }
        }
        let mut prefixes = BTreeSet::new();
        for rule in &raw.labels {
            if rule.documents.is_empty() {
                return Err(InvalidAccessConfig::EmptyPrefix);
            }
            if !prefixes.insert(rule.documents.as_str()) {
                return Err(InvalidAccessConfig::DuplicatePrefix(rule.documents.clone()));
            }
        }
        Ok(Self {
            roles: raw.roles,
            everyone: raw.everyone,
            members: raw.members,
            labels: raw.labels,
        })
    }
}

impl From<AccessConfig> for RawAccessConfig {
    fn from(config: AccessConfig) -> Self {
        Self {
            roles: config.roles,
            everyone: config.everyone,
            members: config.members,
            labels: config.labels,
        }
    }
}

impl AccessConfig {
    /// Reads and parses the file at `path`.
    ///
    /// # Errors
    ///
    /// When the file cannot be read, or as [`AccessConfig::parse`].
    pub fn read(path: &Path) -> eros::Result<Self> {
        read(path)
    }

    /// # Errors
    ///
    /// On invalid JSON, an unknown key, an invalid role name, or an [`InvalidAccessConfig`].
    pub fn parse(text: &str) -> eros::Result<Self> {
        Ok(serde_json::from_str(text)?)
    }

    /// The label rules, for [`LabelRules::new`].
    pub fn labels(&self) -> &[LabelRule] {
        &self.labels
    }

    /// The roles `principal` holds: those it asserts, `everyone`, and its `members` entry.
    fn effective_roles(&self, principal: &Principal) -> BTreeSet<Role> {
        let mut roles = principal.roles().clone();
        roles.extend(self.everyone.iter().cloned());
        if let Some(member) = self.members.get(principal.id()) {
            roles.extend(member.iter().cloned());
        }
        roles
    }
}

#[eros::context("reading access policy {}", path.display())]
fn read(path: &Path) -> eros::Result<AccessConfig> {
    let text = fs::read_to_string(path)?;
    AccessConfig::parse(&text)
}

/// The shipped [`Authorizer`]: a principal's roles are the ones it asserts, `everyone`, and its `members` entry;
/// each role defined in `roles` adds its grant, in role-name order. Roles the policy does not define grant nothing,
/// but still name the principal in ACLs.
#[derive(Clone, Debug)]
pub struct RulePolicy {
    config: AccessConfig,
    version: String,
}

impl RulePolicy {
    /// Decides by `config`; its version is a hash of the config, so a trace names the exact policy.
    pub fn new(config: AccessConfig) -> Self {
        let bytes = serde_json::to_vec(&config)
            .expect("an access config is plain data, which always serializes");
        Self {
            version: hex::encode(Sha256::digest(bytes)),
            config,
        }
    }
}

#[async_trait]
impl Authorizer for RulePolicy {
    fn policy(&self) -> PolicyRef {
        PolicyRef {
            id: "rules".into(),
            version: self.version.clone(),
        }
    }

    async fn access(&self, principal: &Principal) -> eros::Result<Access> {
        let roles = self.config.effective_roles(principal);
        let defined: Vec<(&Role, &RoleGrant)> = roles
            .iter()
            .filter_map(|role| self.config.roles.get_key_value(role))
            .collect();
        let grants = defined.iter().map(|(role, grant)| {
            ChunkGrant::new((*role).clone(), grant.clearance, grant.tenants.clone())
        });
        let members = std::iter::once(principal.id().0.clone())
            .chain(roles.iter().map(|role| role.as_str().to_owned()));
        let runs = defined
            .iter()
            .filter_map(|(role, grant)| {
                let role = (*role).clone();
                match grant.runs {
                    RunGrant::None => None,
                    RunGrant::Own => Some((RunGrant::Own, RunAccess::Own { role })),
                    RunGrant::Any => Some((RunGrant::Any, RunAccess::Any { role })),
                }
            })
            .max_by_key(|(widest, _)| *widest)
            .map_or(RunAccess::Nothing, |(_, runs)| runs);
        Ok(Access::new(ChunkScope::new(members, grants), runs))
    }
}

/// The shipped [`Labeler`]: the longest [`LabelRule::documents`] prefix of a document's id sets its labels; a
/// document no rule matches keeps the connector's.
#[derive(Clone, Debug, Default)]
pub struct LabelRules {
    rules: Vec<LabelRule>,
}

impl LabelRules {
    /// Labels by `rules`.
    pub fn new(rules: &[LabelRule]) -> Self {
        Self {
            rules: rules.to_vec(),
        }
    }
}

#[async_trait]
impl Labeler for LabelRules {
    async fn label(&self, document: &Document) -> eros::Result<Labels> {
        let id = document.id();
        Ok(self
            .rules
            .iter()
            .filter(|rule| id.0.starts_with(&rule.documents))
            .max_by_key(|rule| rule.documents.len())
            .map_or_else(|| document.labels.clone(), LabelRule::labels))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(id: &str) -> Principal {
        Principal::new(UserId(id.into()))
    }

    fn labels(sensitivity: Sensitivity, tenant: Option<&str>, acl: &[&str]) -> Labels {
        Labels {
            acl: acl.iter().map(|entry| (*entry).to_owned()).collect(),
            tenant: tenant.map(str::to_owned),
            sensitivity,
        }
    }

    fn policy(json: &str) -> RulePolicy {
        RulePolicy::new(AccessConfig::parse(json).unwrap())
    }

    const POLICY: &str = r#"{
        "roles": {
            "reader": { "clearance": "internal", "runs": "own" },
            "acme": { "clearance": "internal", "tenants": ["acme"] },
            "maintainer": { "clearance": "confidential", "tenants": "any", "runs": "any" }
        },
        "everyone": ["reader"],
        "members": { "jose": ["maintainer"], "ana": ["acme"] }
    }"#;

    #[test]
    fn the_shipped_policy_parses() {
        let config = AccessConfig::parse(include_str!("../../config/access.json")).unwrap();
        assert!(!config.roles.is_empty());
    }

    #[tokio::test]
    async fn roles_come_from_everyone_members_and_the_principal() {
        let policy = policy(POLICY);
        let internal = labels(Sensitivity::Internal, None, &[]);
        let confidential = labels(Sensitivity::Confidential, None, &[]);
        let acme = labels(Sensitivity::Internal, Some("acme"), &[]);

        let reader = policy.access(&user("eve")).await.unwrap();
        assert!(reader.chunks().admits(&internal));
        assert!(!reader.chunks().admits(&confidential));
        assert!(!reader.chunks().admits(&acme));

        let ana = policy.access(&user("ana")).await.unwrap();
        assert_eq!(
            ana.chunks().decide(&acme),
            super::super::Decision::Allowed {
                role: Role::new("acme").unwrap()
            }
        );

        let jose = policy.access(&user("jose")).await.unwrap();
        assert!(jose.chunks().admits(&confidential));
        assert!(
            jose.chunks()
                .admits(&labels(Sensitivity::Public, Some("x"), &[]))
        );

        let asserted = user("eve").with_roles([Role::new("maintainer").unwrap()]);
        let asserted = policy.access(&asserted).await.unwrap();
        assert!(asserted.chunks().admits(&confidential));
    }

    #[tokio::test]
    async fn an_acl_needs_the_principal_or_one_of_its_roles() {
        let policy = policy(POLICY);
        let jose = policy.access(&user("jose")).await.unwrap();
        assert!(
            !jose
                .chunks()
                .admits(&labels(Sensitivity::Public, None, &["hr"]))
        );
        assert!(
            jose.chunks()
                .admits(&labels(Sensitivity::Public, None, &["hr", "jose"]))
        );
        assert!(
            jose.chunks()
                .admits(&labels(Sensitivity::Public, None, &["maintainer"]))
        );
        let hr = user("eve").with_roles([Role::new("hr").unwrap()]);
        let hr = policy.access(&hr).await.unwrap();
        assert!(
            hr.chunks()
                .admits(&labels(Sensitivity::Public, None, &["hr"]))
        );
    }

    #[tokio::test]
    async fn run_access_is_the_widest_role_grant() {
        let policy = policy(POLICY);
        let eve = user("eve");
        let reader = policy.access(&eve).await.unwrap();
        assert!(
            reader
                .runs()
                .decide(&eve, &UserId("eve".into()))
                .is_allowed()
        );
        assert!(
            !reader
                .runs()
                .decide(&eve, &UserId("ana".into()))
                .is_allowed()
        );
        let jose = user("jose");
        let maintainer = policy.access(&jose).await.unwrap();
        assert!(
            maintainer
                .runs()
                .decide(&jose, &UserId("ana".into()))
                .is_allowed()
        );
    }

    #[tokio::test]
    async fn without_everyone_an_unknown_principal_gets_nothing() {
        let policy = policy(r#"{ "roles": { "reader": { "clearance": "public" } } }"#);
        let access = policy.access(&user("eve")).await.unwrap();
        assert!(access.chunks().grants().is_empty());
        assert_eq!(access.runs(), &RunAccess::Nothing);
        let undefined = user("eve").with_roles([Role::new("root").unwrap()]);
        assert!(
            policy
                .access(&undefined)
                .await
                .unwrap()
                .chunks()
                .grants()
                .is_empty()
        );
    }

    #[test]
    fn undefined_roles_and_bad_label_rules_do_not_load() {
        for bad in [
            r#"{ "roles": {}, "everyone": ["reader"] }"#,
            r#"{ "roles": {}, "members": { "ana": ["reader"] } }"#,
            r#"{ "roles": {}, "labels": [{ "documents": "", "sensitivity": "public" }] }"#,
            r#"{ "roles": {}, "labels": [{ "documents": "a", "sensitivity": "public" },
                                         { "documents": "a", "sensitivity": "internal" }] }"#,
            r#"{ "roles": { "Reader": { "clearance": "public" } } }"#,
            r#"{ "roles": {}, "admins": [] }"#,
            r#"{ "everyone": [] }"#,
        ] {
            assert!(AccessConfig::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_version_changes_with_the_config() {
        let a = policy(POLICY).policy();
        let b = policy(r#"{ "roles": {} }"#).policy();
        assert_eq!(a.id, "rules");
        assert_ne!(a.version, b.version);
        assert_eq!(a, policy(POLICY).policy());
    }

    #[tokio::test]
    async fn the_longest_prefix_labels_a_document() {
        let config = AccessConfig::parse(
            r#"{ "roles": {}, "labels": [
                { "documents": "markdown:hr/", "acl": ["hr"], "sensitivity": "confidential" },
                { "documents": "markdown:hr/public/", "sensitivity": "public" }
            ] }"#,
        )
        .unwrap();
        let rules = LabelRules::new(config.labels());
        let document = |path: &str| Document {
            source: crate::domain::SourceName::from_static("markdown"),
            external_id: path.into(),
            title: path.into(),
            url: String::new(),
            updated_at: chrono::Utc::now(),
            labels: labels(Sensitivity::Internal, None, &[]),
            chunks: vec![],
        };
        assert_eq!(
            rules.label(&document("hr/pay.md")).await.unwrap(),
            labels(Sensitivity::Confidential, None, &["hr"])
        );
        assert_eq!(
            rules.label(&document("hr/public/faq.md")).await.unwrap(),
            labels(Sensitivity::Public, None, &[])
        );
        assert_eq!(
            rules.label(&document("eng/roles.md")).await.unwrap(),
            labels(Sensitivity::Internal, None, &[])
        );
    }
}
