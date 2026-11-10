use std::collections::BTreeSet;

use super::*;

fn role(name: &str) -> Role {
    Role::new(name).unwrap()
}

fn labels(sensitivity: Sensitivity, tenant: Option<&str>, acl: &[&str]) -> Labels {
    Labels {
        acl: acl.iter().map(|entry| (*entry).to_owned()).collect(),
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
        ["ana"],
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
        ["ana", "hr"],
        [ChunkGrant::new(
            role("maintainer"),
            Sensitivity::Confidential,
            TenantScope::Any,
        )],
    );
    assert_eq!(
        scope.decide(&labels(Sensitivity::Public, None, &["finance"])),
        denied(DenyReason::NotInAcl)
    );
    assert!(scope.admits(&labels(Sensitivity::Public, None, &["finance", "hr"])));
    assert!(scope.admits(&labels(Sensitivity::Public, None, &["ana"])));
}

#[test]
fn the_first_admitting_grant_is_named() {
    let scope = ChunkScope::new(
        ["ana"],
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
            ["ana", "reader"],
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
    assert_eq!(serde_json::from_value::<Access>(json).unwrap(), access);
}

#[test]
fn run_access_decides_by_owner() {
    let ana = Principal::new(UserId("ana".into()));
    let mine = ana.id().clone();
    let theirs = UserId("jose".into());
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
        ["ana", "jose"]
            .map(|id| Principal::new(UserId(id.into())))
            .to_vec()
    }

    fn authorizer() -> StaticAuthorizer {
        StaticAuthorizer::new(principals().into_iter().map(|principal| {
            let grant = ChunkGrant::new(
                Role::new("reader").unwrap(),
                Sensitivity::Internal,
                TenantScope::Any,
            );
            let access = Access::new(
                ChunkScope::new([principal.id().0.clone()], [grant]),
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

    fn principals() -> Vec<Principal> {
        vec![
            Principal::new(UserId("ana".into())),
            Principal::new(UserId("jose".into())),
            Principal::new(UserId("eve".into())).with_roles([Role::new("auditor").unwrap()]),
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
                    "members": { "ana": ["reader"], "jose": ["reader", "auditor"] }
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
                { "documents": "markdown:hr/", "acl": ["hr"], "sensitivity": "confidential" },
                { "documents": "markdown:hr/public/", "sensitivity": "public" }
            ] }"#,
        )
        .unwrap();
        LabelRules::new(config.labels())
    }

    crate::labeler_contract!(#[tokio::test] () => rules(), crate::access::contract::documents());
}

#[tokio::test]
async fn failing_adapters_fail() {
    use super::fakes::{FailingAuthorizer, FailingLabeler};
    let principal = Principal::new(UserId("ana".into()));
    assert!(FailingAuthorizer.access(&principal).await.is_err());
    let document = &contract::documents()[0];
    assert!(FailingLabeler.label(document).await.is_err());
}
