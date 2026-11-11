use std::collections::HashMap;

use async_trait::async_trait;

use crate::domain::{Document, Labels, Sensitivity, UserId};

use super::{
    Access, Authorizer, ChunkGrant, ChunkScope, Labeler, PolicyRef, Principal, Role, RunAccess,
    TenantScope,
};

/// The principal tests ask as when access is not their subject.
pub fn tester() -> Principal {
    Principal::new(UserId::new("tester").expect("a valid user id"))
}

/// Every chunk of every tenant with an empty ACL or one naming [`tester`], and every run: for tests whose subject is
/// not access.
pub fn unrestricted() -> Access {
    let role = Role::new("tester").expect("a valid role name");
    Access::new(
        ChunkScope::new(
            tester().id().clone(),
            [],
            [ChunkGrant::new(
                role.clone(),
                Sensitivity::Confidential,
                TenantScope::Any,
            )],
        ),
        RunAccess::Any { role },
    )
}

/// Gives each listed principal its [`Access`]; any other principal gets [`Access::none`], as every adapter must.
#[derive(Clone, Debug, Default)]
pub struct StaticAuthorizer {
    access: HashMap<UserId, Access>,
}

impl StaticAuthorizer {
    /// `access` per principal id; roles the principal asserts are ignored.
    pub fn new(access: impl IntoIterator<Item = (UserId, Access)>) -> Self {
        Self {
            access: access.into_iter().collect(),
        }
    }

    /// [`unrestricted`] access for [`tester`] only.
    pub fn tester() -> Self {
        Self::new([(tester().id().clone(), unrestricted())])
    }
}

#[async_trait]
impl Authorizer for StaticAuthorizer {
    fn policy(&self) -> PolicyRef {
        PolicyRef {
            id: "fake-static".into(),
            version: "1".into(),
        }
    }

    async fn access(&self, principal: &Principal) -> eros::Result<Access> {
        Ok(self.access.get(principal.id()).cloned().unwrap_or_default())
    }
}

/// Fails every call, as an unreachable policy service would, for error-path tests.
#[derive(Clone, Copy, Debug, Default)]
pub struct FailingAuthorizer;

#[async_trait]
impl Authorizer for FailingAuthorizer {
    fn policy(&self) -> PolicyRef {
        PolicyRef {
            id: "fake-failing".into(),
            version: "1".into(),
        }
    }

    async fn access(&self, _: &Principal) -> eros::Result<Access> {
        eros::bail!("policy service at https://policy.internal timed out")
    }
}

/// Fails every call, as an unreachable classification service would, for error-path tests.
#[derive(Clone, Copy, Debug, Default)]
pub struct FailingLabeler;

#[async_trait]
impl Labeler for FailingLabeler {
    async fn label(&self, document: &Document) -> eros::Result<Labels> {
        eros::bail!("labels for {} are unavailable", document.id().0)
    }
}
