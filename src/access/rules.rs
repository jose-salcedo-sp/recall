//! The shipped policy: roles and their grants, role membership and label rules, from `config/access.json`
//! (`RECALL_ACCESS`). [`RulePolicy`] is its [`Authorizer`], [`LabelRules`] its [`Labeler`].
//!
//! `everyone` is a baseline grant: every principal holds those roles, whether or not the policy lists it, so a
//! non-empty `everyone` is allow-by-default for authenticated callers. The shipped file leaves it empty.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt, fs,
    path::Path,
};

use async_trait::async_trait;
use eros::Context;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::domain::{Document, Labels, Sensitivity, SourceName, UserId};

use super::{
    Access, Acl, Authorizer, ChunkGrant, ChunkScope, Labeler, PolicyRef, Principal, Role, RunAccess,
};

/// Where the shipped policy lives unless `RECALL_ACCESS` says otherwise.
pub const DEFAULT_PATH: &str = "config/access.json";

/// What one role may read.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoleGrant {
    /// The most sensitive tier it reads.
    pub clearance: Sensitivity,
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

/// Labels that tighten every document of `source` under the `/`-separated path prefix `path`: the document keeps the
/// connector's labels, narrowed by these ([`Labels::tighten`]). A rule never widens what the connector set.
///
/// Written `{ "source": "markdown", "path": "hr/", "acl": ["role:hr"], "sensitivity": "confidential" }`. `path`
/// matches whole segments of the external id (`hr` covers `hr/pay.md`, not `hrx/pay.md`) and covers the whole
/// source when empty; `acl` defaults to no restriction and `sensitivity` to `public`, which changes
/// nothing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "RawLabelRule", into = "RawLabelRule")]
pub struct LabelRule {
    source: SourceName,
    path: String,
    labels: Labels,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLabelRule {
    source: SourceName,
    #[serde(default)]
    path: String,
    #[serde(default)]
    acl: Acl,
    #[serde(default = "least_sensitive")]
    sensitivity: Sensitivity,
}

fn least_sensitive() -> Sensitivity {
    Sensitivity::Public
}

impl From<RawLabelRule> for LabelRule {
    fn from(raw: RawLabelRule) -> Self {
        Self {
            source: raw.source,
            path: raw.path.trim_end_matches('/').to_owned(),
            labels: Labels {
                acl: raw.acl,
                sensitivity: raw.sensitivity,
            },
        }
    }
}

impl From<LabelRule> for RawLabelRule {
    fn from(rule: LabelRule) -> Self {
        Self {
            source: rule.source,
            path: rule.path,
            acl: rule.labels.acl,
            sensitivity: rule.labels.sensitivity,
        }
    }
}

impl LabelRule {
    fn covers(&self, document: &Document) -> bool {
        document.source == self.source
            && (self.path.is_empty()
                || document
                    .external_id
                    .strip_prefix(&self.path)
                    .is_some_and(|rest| rest.is_empty() || rest.starts_with('/')))
    }
}

impl fmt::Display for LabelRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.source, self.path)
    }
}

/// The parsed policy. Every role named in `everyone` or `members` is defined in `roles`, and no two label rules
/// cover the same source and path; unknown keys are rejected.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "RawAccessConfig", into = "RawAccessConfig")]
pub struct AccessConfig {
    roles: BTreeMap<Role, RoleGrant>,
    everyone: BTreeSet<Role>,
    members: BTreeMap<UserId, BTreeSet<Role>>,
    labels: LabelRules,
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
    labels: LabelRules,
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
}

/// Two label rules for one source and path.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[error("two label rules cover {0}")]
pub struct DuplicateLabelRule(pub String);

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
            if let Some(error) = undefined(roles, &|| format!("members.{user}")) {
                return Err(error);
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

    /// The label rules, the policy's [`Labeler`].
    pub fn labels(&self) -> &LabelRules {
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
        PolicyRef::new("rules", self.version.clone())
            .expect("the version is a hex digest, never empty")
    }

    async fn access(&self, principal: &Principal) -> eros::Result<Access> {
        let roles = self.config.effective_roles(principal);
        let defined: Vec<(&Role, &RoleGrant)> = roles
            .iter()
            .filter_map(|role| self.config.roles.get_key_value(role))
            .collect();
        let grants = defined
            .iter()
            .map(|(role, grant)| ChunkGrant::new((*role).clone(), grant.clearance));
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
        let chunks = ChunkScope::new(principal.id().clone(), roles.iter().cloned(), grants);
        Ok(Access::new(chunks, runs))
    }
}

/// The shipped [`Labeler`]: every [`LabelRule`] that covers a document tightens the connector's labels, so adding a
/// rule never widens access. No two rules cover the same source and path.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "Vec<LabelRule>", into = "Vec<LabelRule>")]
pub struct LabelRules {
    rules: Vec<LabelRule>,
}

impl LabelRules {
    /// # Errors
    ///
    /// [`DuplicateLabelRule`] when two rules cover the same source and path.
    pub fn new(rules: Vec<LabelRule>) -> Result<Self, DuplicateLabelRule> {
        let mut covered = BTreeSet::new();
        if let Some(duplicate) = rules
            .iter()
            .find(|rule| !covered.insert((&rule.source, &rule.path)))
        {
            return Err(DuplicateLabelRule(duplicate.to_string()));
        }
        Ok(Self { rules })
    }
}

impl TryFrom<Vec<LabelRule>> for LabelRules {
    type Error = DuplicateLabelRule;

    fn try_from(rules: Vec<LabelRule>) -> Result<Self, DuplicateLabelRule> {
        Self::new(rules)
    }
}

impl From<LabelRules> for Vec<LabelRule> {
    fn from(rules: LabelRules) -> Self {
        rules.rules
    }
}

#[async_trait]
impl Labeler for LabelRules {
    async fn label(&self, document: &Document) -> eros::Result<Labels> {
        let mut labels = document.labels.clone();
        for rule in self.rules.iter().filter(|rule| rule.covers(document)) {
            labels = labels.tighten(&rule.labels).with_context(|| {
                format!("label rule {rule} conflicts with the connector's labels")
            })?;
        }
        Ok(labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access::{AclEntry, Decision};

    fn user(id: &str) -> Principal {
        Principal::new(UserId::new(id).unwrap())
    }

    fn acl(entries: &[&str]) -> Acl {
        Acl::only(
            entries
                .iter()
                .map(|entry| entry.parse::<AclEntry>().unwrap()),
        )
    }

    fn labels(sensitivity: Sensitivity, entries: &[&str]) -> Labels {
        Labels {
            acl: acl(entries),
            sensitivity,
        }
    }

    fn policy(json: &str) -> RulePolicy {
        RulePolicy::new(AccessConfig::parse(json).unwrap())
    }

    const POLICY: &str = r#"{
        "roles": {
            "reader": { "clearance": "internal", "runs": "own" },
            "auditor": { "clearance": "confidential" },
            "maintainer": { "clearance": "confidential", "runs": "any" }
        },
        "everyone": ["reader"],
        "members": { "jose": ["maintainer"], "ana": ["auditor"] }
    }"#;

    #[tokio::test]
    async fn the_shipped_policy_denies_a_user_it_does_not_list() {
        let config = AccessConfig::parse(include_str!("../../config/access.json")).unwrap();
        assert!(!config.roles.is_empty());
        let access = RulePolicy::new(config).access(&user("eve")).await.unwrap();
        assert!(access.chunks().grants().is_empty());
        assert_eq!(access.runs(), &RunAccess::Nothing);
    }

    #[tokio::test]
    async fn roles_come_from_everyone_members_and_the_principal() {
        let policy = policy(POLICY);
        let internal = labels(Sensitivity::Internal, &[]);
        let confidential = labels(Sensitivity::Confidential, &[]);

        let reader = policy.access(&user("eve")).await.unwrap();
        assert!(reader.chunks().admits(&internal));
        assert!(!reader.chunks().admits(&confidential));

        let ana = policy.access(&user("ana")).await.unwrap();
        assert_eq!(
            ana.chunks().decide(&confidential),
            Decision::Allowed {
                role: Role::new("auditor").unwrap()
            }
        );
        assert_eq!(
            ana.runs(),
            &RunAccess::Own {
                role: Role::new("reader").unwrap()
            }
        );

        let jose = policy.access(&user("jose")).await.unwrap();
        assert!(jose.chunks().admits(&confidential));

        let asserted = user("eve").with_roles([Role::new("maintainer").unwrap()]);
        let asserted = policy.access(&asserted).await.unwrap();
        assert!(asserted.chunks().admits(&confidential));
    }

    #[tokio::test]
    async fn an_acl_needs_the_user_or_one_of_its_roles() {
        let policy = policy(POLICY);
        let jose = policy.access(&user("jose")).await.unwrap();
        let public = |entries: &[&str]| labels(Sensitivity::Public, entries);
        assert!(!jose.chunks().admits(&public(&["role:hr"])));
        assert!(jose.chunks().admits(&public(&["role:hr", "user:jose"])));
        assert!(jose.chunks().admits(&public(&["role:maintainer"])));
        assert!(!jose.chunks().admits(&public(&["user:ana"])));
        let hr = user("eve").with_roles([Role::new("hr").unwrap()]);
        let hr = policy.access(&hr).await.unwrap();
        assert!(hr.chunks().admits(&public(&["role:hr"])));
    }

    #[tokio::test]
    async fn a_user_whose_id_is_a_role_name_does_not_hold_the_role() {
        let policy = policy(POLICY);
        let restricted = labels(Sensitivity::Public, &["role:maintainer"]);
        let named_maintainer = policy.access(&user("maintainer")).await.unwrap();
        assert!(
            named_maintainer
                .chunks()
                .admits(&labels(Sensitivity::Internal, &[]))
        );
        assert_eq!(
            named_maintainer.chunks().decide(&restricted),
            Decision::Denied {
                reason: crate::access::DenyReason::NotInAcl
            }
        );
        assert!(
            policy
                .access(&user("jose"))
                .await
                .unwrap()
                .chunks()
                .admits(&restricted)
        );
        assert!(
            named_maintainer
                .chunks()
                .admits(&labels(Sensitivity::Public, &["user:maintainer"]))
        );
    }

    #[tokio::test]
    async fn run_access_is_the_widest_role_grant() {
        let policy = policy(POLICY);
        let eve = user("eve");
        let ana = UserId::new("ana").unwrap();
        let reader = policy.access(&eve).await.unwrap();
        assert!(reader.runs().decide(&eve, eve.id()).is_allowed());
        assert!(!reader.runs().decide(&eve, &ana).is_allowed());
        let jose = user("jose");
        let maintainer = policy.access(&jose).await.unwrap();
        assert!(maintainer.runs().decide(&jose, &ana).is_allowed());
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
            r#"{ "roles": {}, "members": { "": [] } }"#,
            r#"{ "roles": {}, "labels": [{ "source": "markdown", "path": "a", "sensitivity": "public" },
                                         { "source": "markdown", "path": "a/", "sensitivity": "internal" }] }"#,
            r#"{ "roles": {}, "labels": [{ "source": "markdown", "acl": ["hr"] }] }"#,
            r#"{ "roles": {}, "labels": [{ "source": "markdown", "acl": ["group:hr"] }] }"#,
            r#"{ "roles": {}, "labels": [{ "source": "markdown", "acl": ["user:"] }] }"#,
            r#"{ "roles": {}, "labels": [{ "source": "Markdown" }] }"#,
            r#"{ "roles": {}, "labels": [{ "documents": "markdown:hr/", "sensitivity": "public" }] }"#,
            r#"{ "roles": { "Reader": { "clearance": "public" } } }"#,
            r#"{ "roles": {}, "admins": [] }"#,
            r#"{ "roles": { "reader": { "clearance": "public", "tenants": "any" } } }"#,
            r#"{ "roles": {}, "labels": [{ "source": "markdown", "tenant": "acme" }] }"#,
            r#"{ "everyone": [] }"#,
        ] {
            assert!(AccessConfig::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_version_changes_with_the_config() {
        let a = policy(POLICY).policy();
        let b = policy(r#"{ "roles": {} }"#).policy();
        assert_eq!(a.id(), "rules");
        assert_ne!(a.version(), b.version());
        assert_eq!(a, policy(POLICY).policy());
    }

    fn rules(json: &str) -> LabelRules {
        AccessConfig::parse(&format!(r#"{{ "roles": {{}}, "labels": {json} }}"#))
            .unwrap()
            .labels()
            .clone()
    }

    fn document(source: &'static str, path: &str, labels: Labels) -> Document {
        Document {
            source: SourceName::from_static(source),
            external_id: path.into(),
            title: path.into(),
            url: String::new(),
            updated_at: chrono::Utc::now(),
            labels,
            chunks: vec![],
        }
    }

    #[tokio::test]
    async fn every_covering_rule_tightens_the_connector_labels() {
        let rules = rules(
            r#"[
                { "source": "markdown", "path": "hr/", "acl": ["role:hr"], "sensitivity": "confidential" },
                { "source": "markdown", "path": "hr/public", "sensitivity": "public" }
            ]"#,
        );
        let internal = labels(Sensitivity::Internal, &[]);
        let hr = labels(Sensitivity::Confidential, &["role:hr"]);
        let label = async |source, path| {
            rules
                .label(&document(source, path, internal.clone()))
                .await
                .unwrap()
        };
        assert_eq!(label("markdown", "hr/pay.md").await, hr);
        assert_eq!(label("markdown", "hr/public/faq.md").await, hr);
        assert_eq!(label("markdown", "hr").await, hr);
        assert_eq!(label("markdown", "hrx/pay.md").await, internal);
        assert_eq!(label("markdown", "eng/hr/pay.md").await, internal);
        assert_eq!(label("wiki", "hr/pay.md").await, internal);
    }

    #[tokio::test]
    async fn a_rule_never_drops_the_connector_acl_or_lowers_its_sensitivity() {
        let rules = rules(
            r#"[{ "source": "wiki", "acl": ["role:hr", "role:finance"], "sensitivity": "internal" }]"#,
        );
        let inherited = labels(Sensitivity::Confidential, &["role:hr", "user:ana"]);
        assert_eq!(
            rules
                .label(&document("wiki", "pay", inherited))
                .await
                .unwrap(),
            labels(Sensitivity::Confidential, &["role:hr"])
        );
        let disjoint = labels(Sensitivity::Public, &["user:ana"]);
        assert!(
            rules
                .label(&document("wiki", "pay", disjoint))
                .await
                .is_err()
        );
    }
}
