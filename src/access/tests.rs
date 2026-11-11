use std::collections::BTreeSet;

use super::*;

fn role(name: &str) -> Role {
    Role::new(name).unwrap()
}

fn user(id: &str) -> UserId {
    UserId::new(id).unwrap()
}

fn acl(entries: &[&str]) -> Acl {
    Acl::only(
        entries
            .iter()
            .map(|entry| entry.parse::<AclEntry>().unwrap()),
    )
}

fn labels(sensitivity: Sensitivity, tenant: Option<&str>, entries: &[&str]) -> Labels {
    Labels {
        acl: acl(entries),
        tenant: tenant.map(str::to_owned),
        sensitivity,
    }
}

fn denied(reason: DenyReason) -> Decision {
    Decision::Denied { reason }
}

#[test]
fn roles_are_validated() {
    for good in ["reader", "hr:payroll", "team.eng-1_a"] {
        assert_eq!(Role::new(good).unwrap().as_str(), good);
    }
    for bad in ["", "Reader", "a b", "x/y"] {
        assert_eq!(Role::new(bad), Err(InvalidRole(bad.into())));
    }
    assert!(serde_json::from_str::<Role>("\"A\"").is_err());
}

#[test]
fn no_scope_admits_nothing() {
    for sensitivity in Sensitivity::ALL {
        assert_eq!(
            ChunkScope::none().decide(&labels(sensitivity, None, &[])),
            denied(DenyReason::NoGrant)
        );
    }
}

#[test]
fn a_grant_admits_up_to_its_clearance_and_its_tenants() {
    let scope = ChunkScope::new(
        user("ana"),
        [],
        [ChunkGrant::new(
            role("reader"),
            Sensitivity::Internal,
            TenantScope::Only(BTreeSet::from(["acme".to_owned()])),
        )],
    );
    assert!(scope.admits(&labels(Sensitivity::Public, None, &[])));
    assert!(scope.admits(&labels(Sensitivity::Internal, Some("acme"), &[])));
    assert_eq!(
        scope.decide(&labels(Sensitivity::Confidential, None, &[])),
        denied(DenyReason::NoGrant)
    );
    assert_eq!(
        scope.decide(&labels(Sensitivity::Internal, Some("globex"), &[])),
        denied(DenyReason::NoGrant)
    );
}

#[test]
fn the_acl_is_checked_before_any_grant() {
    let scope = ChunkScope::new(
        user("ana"),
        [role("hr")],
        [ChunkGrant::new(
            role("maintainer"),
            Sensitivity::Confidential,
            TenantScope::Any,
        )],
    );
    assert_eq!(
        scope.decide(&labels(Sensitivity::Public, None, &["role:finance"])),
        denied(DenyReason::NotInAcl)
    );
    assert!(scope.admits(&labels(
        Sensitivity::Public,
        None,
        &["role:finance", "role:hr"]
    )));
    assert!(scope.admits(&labels(Sensitivity::Public, None, &["user:ana"])));
    assert_eq!(
        scope.decide(&labels(Sensitivity::Public, None, &["user:hr", "role:ana"])),
        denied(DenyReason::NotInAcl)
    );
}

#[test]
fn acl_entries_are_tagged_users_or_roles() {
    assert_eq!(
        "user:Ana Smith".parse::<AclEntry>().unwrap(),
        AclEntry::User(user("Ana Smith"))
    );
    assert_eq!(
        "role:hr:payroll".parse::<AclEntry>().unwrap(),
        AclEntry::Role(role("hr:payroll"))
    );
    for bad in [
        "hr",
        "",
        "user:",
        "role:",
        "role:HR",
        "group:hr",
        "user:a\nb",
    ] {
        assert!(bad.parse::<AclEntry>().is_err(), "{bad}");
    }
    let entries = acl(&["user:hr", "role:hr"]);
    assert_eq!(entries.entries().len(), 2);
    let json = serde_json::to_value(&entries).unwrap();
    assert_eq!(json, serde_json::json!(["user:hr", "role:hr"]));
    assert_eq!(serde_json::from_value::<Acl>(json).unwrap(), entries);
    assert!(serde_json::from_value::<Acl>(serde_json::json!(["hr"])).is_err());
}

#[test]
fn tightening_an_acl_never_widens_it() {
    let hr = acl(&["role:hr", "user:ana"]);
    assert_eq!(Acl::anyone().tighten(&hr).unwrap(), hr);
    assert_eq!(hr.tighten(&Acl::anyone()).unwrap(), hr);
    assert_eq!(
        hr.tighten(&acl(&["role:hr", "role:finance"])).unwrap(),
        acl(&["role:hr"])
    );
    assert!(hr.tighten(&acl(&["role:finance"])).is_err());
}

#[test]
fn a_stored_scope_naming_two_users_does_not_load() {
    let json = serde_json::json!({ "members": ["user:ana", "user:jose"], "grants": [] });
    assert!(serde_json::from_value::<ChunkScope>(json).is_err());
}

#[test]
fn the_first_admitting_grant_is_named() {
    let scope = ChunkScope::new(
        user("ana"),
        [],
        [
            ChunkGrant::new(role("reader"), Sensitivity::Public, TenantScope::default()),
            ChunkGrant::new(role("staff"), Sensitivity::Internal, TenantScope::Any),
        ],
    );
    assert_eq!(
        scope.decide(&labels(Sensitivity::Public, None, &[])),
        Decision::Allowed {
            role: role("reader")
        }
    );
    assert_eq!(
        scope.decide(&labels(Sensitivity::Public, Some("acme"), &[])),
        Decision::Allowed {
            role: role("staff")
        }
    );
}

#[test]
fn tenant_scope_is_any_or_a_list_in_json() {
    assert_eq!(
        serde_json::from_str::<TenantScope>("\"any\"").unwrap(),
        TenantScope::Any
    );
    assert_eq!(
        serde_json::from_str::<TenantScope>("[\"acme\"]").unwrap(),
        TenantScope::Only(BTreeSet::from(["acme".to_owned()]))
    );
    assert!(serde_json::from_str::<TenantScope>("\"all\"").is_err());
    for scope in [TenantScope::Any, TenantScope::default()] {
        let json = serde_json::to_string(&scope).unwrap();
        assert_eq!(serde_json::from_str::<TenantScope>(&json).unwrap(), scope);
    }
}

#[test]
fn access_round_trips_through_json() {
    let access = Access::new(
        ChunkScope::new(
            user("ana"),
            [role("reader")],
            [ChunkGrant::new(
                role("reader"),
                Sensitivity::Internal,
                TenantScope::Any,
            )],
        ),
        RunAccess::Own {
            role: role("reader"),
        },
    );
    let json = serde_json::to_value(&access).unwrap();
    assert_eq!(json["runs"]["runs"], "own");
    assert_eq!(
        json["chunks"]["members"],
        serde_json::json!(["user:ana", "role:reader"])
    );
    assert_eq!(serde_json::from_value::<Access>(json).unwrap(), access);
}

#[test]
fn run_access_decides_by_owner() {
    let ana = Principal::new(user("ana"));
    let mine = ana.id().clone();
    let theirs = user("jose");
    let own = RunAccess::Own {
        role: role("reader"),
    };
    assert!(own.decide(&ana, &mine).is_allowed());
    assert_eq!(own.decide(&ana, &theirs), denied(DenyReason::NotOwner));
    assert_eq!(
        RunAccess::Nothing.decide(&ana, &mine),
        denied(DenyReason::NoGrant)
    );
    assert!(
        RunAccess::Any {
            role: role("auditor")
        }
        .decide(&ana, &theirs)
        .is_allowed()
    );
}

mod static_authorizer {
    use crate::{
        access::{
            Access, ChunkGrant, ChunkScope, Principal, Role, TenantScope, fakes::StaticAuthorizer,
        },
        domain::{Sensitivity, UserId},
    };

    fn principals() -> Vec<Principal> {
        ["ana", "jose", "staff"]
            .map(|id| Principal::new(UserId::new(id).unwrap()))
            .to_vec()
    }

    /// Everyone but the user `staff` holds the role `staff`.
    fn authorizer() -> StaticAuthorizer {
        StaticAuthorizer::new(principals().into_iter().map(|principal| {
            let staff = Role::new("staff").unwrap();
            let roles = (principal.id().as_str() != staff.as_str()).then_some(staff);
            let grant = ChunkGrant::new(
                Role::new("reader").unwrap(),
                Sensitivity::Internal,
                TenantScope::Any,
            );
            let access = Access::new(
                ChunkScope::new(principal.id().clone(), roles, [grant]),
                Default::default(),
            );
            (principal.id().clone(), access)
        }))
    }

    crate::authorizer_contract!(#[tokio::test] () => authorizer(), principals());
}

mod rule_policy {
    use crate::{
        access::{
            Principal, Role,
            rules::{AccessConfig, RulePolicy},
        },
        domain::UserId,
    };

    /// `auditor` is both a role jose holds and the id of a user who does not hold it.
    fn principals() -> Vec<Principal> {
        let user = |id: &str| Principal::new(UserId::new(id).unwrap());
        vec![
            user("ana"),
            user("jose"),
            user("eve").with_roles([Role::new("auditor").unwrap()]),
            user("auditor"),
        ]
    }

    fn policy() -> RulePolicy {
        RulePolicy::new(
            AccessConfig::parse(
                r#"{
                    "roles": {
                        "reader": { "clearance": "internal", "runs": "own" },
                        "auditor": { "clearance": "confidential", "tenants": "any", "runs": "any" }
                    },
                    "members": { "ana": ["reader"], "jose": ["reader", "auditor"], "auditor": ["reader"] }
                }"#,
            )
            .unwrap(),
        )
    }

    crate::authorizer_contract!(#[tokio::test] () => policy(), principals());
}

mod connector_labels {
    crate::labeler_contract!(
        #[tokio::test] () => crate::access::ConnectorLabels,
        crate::access::contract::documents()
    );
}

mod label_rules {
    use crate::access::rules::{AccessConfig, LabelRules};

    fn rules() -> LabelRules {
        let config = AccessConfig::parse(
            r#"{ "roles": {}, "labels": [
                { "source": "markdown", "path": "hr/", "acl": ["role:hr"], "sensitivity": "confidential" },
                { "source": "markdown", "path": "hr/public/", "sensitivity": "public" }
            ] }"#,
        )
        .unwrap();
        config.labels().clone()
    }

    crate::labeler_contract!(#[tokio::test] () => rules(), crate::access::contract::documents());
}

#[tokio::test]
async fn failing_adapters_fail() {
    use super::fakes::{FailingAuthorizer, FailingLabeler};
    let principal = Principal::new(user("ana"));
    assert!(FailingAuthorizer.access(&principal).await.is_err());
    let document = &contract::documents()[0];
    assert!(FailingLabeler.label(document).await.is_err());
}
