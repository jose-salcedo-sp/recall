use std::{
    collections::{HashMap, hash_map::Entry},
    sync::Mutex,
};

use async_trait::async_trait;
use chrono::Utc;

use crate::{
    domain::{RunId, UserId},
    pipeline::report::RunReport,
};

use super::{RunStore, StoredRun};

/// Keeps runs in a map; like the Postgres store, rejects a second save of the same run id.
#[derive(Default)]
pub struct MemoryRunStore {
    runs: Mutex<HashMap<RunId, StoredRun>>,
}

#[async_trait]
impl RunStore for MemoryRunStore {
    async fn save(&self, user: &UserId, report: &RunReport) -> eros::Result<()> {
        let mut runs = self
            .runs
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it");
        let Entry::Vacant(slot) = runs.entry(report.id) else {
            eros::bail!("run {} already saved", report.id);
        };
        slot.insert(StoredRun {
            user: user.clone(),
            created_at: Utc::now(),
            report: report.clone(),
        });
        Ok(())
    }

    async fn get(&self, id: RunId) -> eros::Result<Option<StoredRun>> {
        Ok(self
            .runs
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it")
            .get(&id)
            .cloned())
    }
}

/// Fails every call, with a message like a database error's, for error-path tests.
pub struct FailingRunStore;

#[async_trait]
impl RunStore for FailingRunStore {
    async fn save(&self, _: &UserId, _: &RunReport) -> eros::Result<()> {
        eros::bail!("pool timed out connecting to postgres://drew@db.internal:5432/drew")
    }

    async fn get(&self, _: RunId) -> eros::Result<Option<StoredRun>> {
        eros::bail!("pool timed out connecting to postgres://drew@db.internal:5432/drew")
    }
}

impl MemoryRunStore {
    /// Every saved run, in no particular order.
    pub fn stored(&self) -> Vec<StoredRun> {
        self.runs
            .lock()
            .expect("lock poisoned: a test thread panicked while holding it")
            .values()
            .cloned()
            .collect()
    }
}
