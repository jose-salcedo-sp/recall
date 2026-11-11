//! The rules every [`Authorizer`] and [`Labeler`] keeps, as one async check per rule. [`authorizer_contract!`] and
//! [`labeler_contract!`] turn them into tests; each adapter, the fakes included, calls its macro once.
//!
//! ```text
//! recall::authorizer_contract!(#[tokio::test] () => MyPolicy::new(fixture_config()), known_principals());
//! recall::labeler_contract!(#[tokio::test] () => MyLabeler::new(), fixture_documents());
//! ```
//!
//! [`authorizer_contract!`]: crate::authorizer_contract
//! [`labeler_contract!`]: crate::labeler_contract

use chrono::{TimeZone, Utc};

use super::{Acl, AclEntry, Authorizer, Labeler, Principal, Role};
use crate::domain::{ChunkDraft, Document, Labels, Sensitivity, SourceName, UserId};

/// Labels across every sensitivity, under each ACL in `acls`.
fn label_shapes(acls: &[Acl]) -> Vec<Labels> {
    let mut shapes = Vec::new();
    for sensitivity in Sensitivity::ALL {
        for acl in acls {
            shapes.push(Labels {
                acl: acl.clone(),
                sensitivity,
            });
        }
    }
    shapes
}

fn user(id: &str) -> UserId {
    UserId::new(id).expect("contract fixtures use valid user ids")
}

/// A principal no policy has heard of; one id per call, so it cannot collide with configured principals.
fn stranger() -> Principal {
    Principal::new(user(&format!("stranger-{}", uuid::Uuid::now_v7())))
}

/// Deny by default: a principal the policy does not know reads no chunk and no run, its own included.
///
/// # Errors
///
/// When the authorizer fails.
pub async fn an_unknown_principal_reads_nothing(authorizer: &dyn Authorizer) -> eros::Result<()> {
    let principal = stranger();
    let access = authorizer.access(&principal).await?;
    let nobody_else = Acl::only([AclEntry::User(user("nobody-else"))]);
    for labels in label_shapes(&[Acl::anyone(), nobody_else]) {
        assert!(!access.chunks().admits(&labels), "{labels:?}");
    }
    assert!(
        !access
            .runs()
            .decide(&principal, principal.id())
            .is_allowed()
    );
    assert!(
        !access
            .runs()
            .decide(&principal, &user("someone"))
            .is_allowed()
    );
    Ok(())
}

/// The same principal gets the same access every time, whoever was asked about in between: no state leaks between
/// principals.
///
/// # Errors
///
/// When the authorizer fails.
pub async fn access_depends_only_on_the_principal(
    authorizer: &dyn Authorizer,
    principals: &[Principal],
) -> eros::Result<()> {
    let mut first = Vec::new();
    for principal in principals {
        first.push(authorizer.access(principal).await?);
    }
    authorizer.access(&stranger()).await?;
    for (principal, first) in principals.iter().rev().zip(first.iter().rev()) {
        assert_eq!(&authorizer.access(principal).await?, first, "{principal:?}");
    }
    Ok(())
}

/// A principal's ACL members never name another principal, so one principal cannot read through another's ACL
/// entries.
///
/// # Errors
///
/// When the authorizer fails.
pub async fn acl_members_never_name_another_principal(
    authorizer: &dyn Authorizer,
    principals: &[Principal],
) -> eros::Result<()> {
    for principal in principals {
        let access = authorizer.access(principal).await?;
        for other in principals
            .iter()
            .filter(|other| other.id() != principal.id())
        {
            assert!(
                !access
                    .chunks()
                    .members()
                    .contains(&AclEntry::User(other.id().clone())),
                "{principal:?} is a member as {other:?}"
            );
        }
    }
    Ok(())
}

/// A user id never matches a role entry: a principal whose id is the name of a role another principal holds reads
/// nothing ACL'd to that role. The fixture's `principals` should include such a principal, and the policy must not
/// give it that role.
///
/// # Errors
///
/// When the authorizer fails.
pub async fn a_user_id_never_reads_as_a_role(
    authorizer: &dyn Authorizer,
    principals: &[Principal],
) -> eros::Result<()> {
    let mut held = Vec::new();
    for principal in principals {
        let access = authorizer.access(principal).await?;
        held.extend(
            access
                .chunks()
                .members()
                .iter()
                .filter_map(|entry| match entry {
                    AclEntry::Role(role) => Some((principal.id().clone(), role.clone())),
                    AclEntry::User(_) => None,
                }),
        );
    }
    for principal in principals {
        let Ok(namesake) = Role::new(principal.id().as_str()) else {
            continue;
        };
        if !held
            .iter()
            .any(|(holder, role)| holder != principal.id() && *role == namesake)
        {
            continue;
        }
        let access = authorizer.access(principal).await?;
        let role_entry = AclEntry::Role(namesake);
        for labels in label_shapes(&[Acl::only([role_entry.clone()])]) {
            assert!(
                !access.chunks().admits(&labels),
                "{principal:?} reads as {role_entry} without holding it: {labels:?}"
            );
        }
    }
    Ok(())
}

/// The policy ref does not change between calls.
///
/// # Errors
///
/// Never; the signature matches the other checks.
pub async fn the_policy_ref_is_stable(authorizer: &dyn Authorizer) -> eros::Result<()> {
    let policy = authorizer.policy();
    authorizer.access(&stranger()).await?;
    assert_eq!(authorizer.policy(), policy);
    Ok(())
}

/// One test per authorizer check, for the authorizer `$make` builds and the principals `$principals` lists, which
/// should include principals the policy grants something. The caller gives the test attribute and any arguments the
/// harness passes.
#[macro_export]
macro_rules! authorizer_contract {
    (#[$attr:meta] ($($arg:ident: $ty:ty),*) => $make:expr, $principals:expr) => {
        #[$attr]
        async fn an_unknown_principal_reads_nothing($($arg: $ty),*) -> eros::Result<()> {
            $crate::access::contract::an_unknown_principal_reads_nothing(&$make).await
        }

        #[$attr]
        async fn access_depends_only_on_the_principal($($arg: $ty),*) -> eros::Result<()> {
            $crate::access::contract::access_depends_only_on_the_principal(&$make, &$principals).await
        }

        #[$attr]
        async fn acl_members_never_name_another_principal($($arg: $ty),*) -> eros::Result<()> {
            $crate::access::contract::acl_members_never_name_another_principal(&$make, &$principals)
                .await
        }

        #[$attr]
        async fn a_user_id_never_reads_as_a_role($($arg: $ty),*) -> eros::Result<()> {
            $crate::access::contract::a_user_id_never_reads_as_a_role(&$make, &$principals).await
        }

        #[$attr]
        async fn the_policy_ref_is_stable($($arg: $ty),*) -> eros::Result<()> {
            $crate::access::contract::the_policy_ref_is_stable(&$make).await
        }
    };
}

/// Documents of several sources and paths, labelled the way a connector would, for [`labeler_contract!`].
///
/// [`labeler_contract!`]: crate::labeler_contract
pub fn documents() -> Vec<Document> {
    let document = |source: &'static str, path: &str, sensitivity| Document {
        source: SourceName::from_static(source),
        external_id: path.into(),
        title: path.into(),
        url: format!("https://docs.example.test/{path}"),
        updated_at: Utc
            .with_ymd_and_hms(2026, 1, 1, 0, 0, 0)
            .single()
            .expect("a valid date"),
        labels: Labels {
            acl: Acl::anyone(),
            sensitivity,
        },
        chunks: vec![ChunkDraft {
            locator: "intro".into(),
            heading: None,
            url: format!("https://docs.example.test/{path}#intro"),
            text: format!("About {path}."),
            standing: crate::domain::Standing::Authoritative,
        }],
    };
    vec![
        document("markdown", "hr/pay.md", Sensitivity::Internal),
        document("markdown", "hr/public/faq.md", Sensitivity::Internal),
        document("markdown", "eng/roles.md", Sensitivity::Public),
        document("wiki", "hr/pay", Sensitivity::Confidential),
    ]
}

/// Labels depend only on the document: labelling twice, in either order, gives the same labels.
///
/// # Errors
///
/// When the labeler fails.
pub async fn labels_depend_only_on_the_document(
    labeler: &dyn Labeler,
    documents: &[Document],
) -> eros::Result<()> {
    let mut first = Vec::new();
    for document in documents {
        first.push(labeler.label(document).await?);
    }
    for (document, first) in documents.iter().rev().zip(first.iter().rev()) {
        assert_eq!(
            &labeler.label(document).await?,
            first,
            "{}",
            document.id().as_str()
        );
    }
    Ok(())
}

/// One test per labeler check, for the labeler `$make` builds over the documents `$documents` lists.
#[macro_export]
macro_rules! labeler_contract {
    (#[$attr:meta] ($($arg:ident: $ty:ty),*) => $make:expr, $documents:expr) => {
        #[$attr]
        async fn labels_depend_only_on_the_document($($arg: $ty),*) -> eros::Result<()> {
            $crate::access::contract::labels_depend_only_on_the_document(&$make, &$documents).await
        }
    };
}
