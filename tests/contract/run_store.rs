use drew::{
    domain::{RunId, UserId},
    store::RunStore,
};

use crate::runs::reports;

pub async fn every_outcome_round_trips(store: &dyn RunStore) -> eros::Result<()> {
    let user = UserId("drew".into());
    for (report, _) in reports() {
        store.save(&user, &report).await?;

        let stored = store.get(report.id).await?.expect("the run was saved");
        assert_eq!(stored.user, user);
        assert_eq!(stored.report, report);
    }
    Ok(())
}

pub async fn an_unknown_id_is_none(store: &dyn RunStore) -> eros::Result<()> {
    let (report, _) = reports().remove(0);
    store.save(&UserId("drew".into()), &report).await?;

    assert!(store.get(RunId::new()).await?.is_none());
    Ok(())
}

pub async fn a_duplicate_id_is_rejected_and_the_first_run_kept(
    store: &dyn RunStore,
) -> eros::Result<()> {
    let mut runs = reports();
    let (first, _) = runs.remove(0);
    let (mut second, _) = runs.remove(0);
    second.id = first.id;
    store.save(&UserId("drew".into()), &first).await?;

    let duplicate = store.save(&UserId("jose".into()), &second).await;

    assert!(duplicate.is_err());
    let stored = store.get(first.id).await?.expect("the first run was saved");
    assert_eq!(stored.user, UserId("drew".into()));
    assert_eq!(stored.report, first);
    Ok(())
}

macro_rules! run_store_contract {
    (#[$attr:meta] ($($arg:ident: $ty:ty),*) => $make:expr) => {
        #[$attr]
        async fn every_outcome_round_trips($($arg: $ty),*) -> eros::Result<()> {
            crate::run_store::every_outcome_round_trips(&$make).await
        }

        #[$attr]
        async fn an_unknown_id_is_none($($arg: $ty),*) -> eros::Result<()> {
            crate::run_store::an_unknown_id_is_none(&$make).await
        }

        #[$attr]
        async fn a_duplicate_id_is_rejected_and_the_first_run_kept($($arg: $ty),*) -> eros::Result<()> {
            crate::run_store::a_duplicate_id_is_rejected_and_the_first_run_kept(&$make).await
        }
    };
}

mod memory {
    use drew::store::fakes::MemoryRunStore;

    run_store_contract!(#[tokio::test] () => MemoryRunStore::default());
}

mod postgres {
    use drew::store::runs::PgRunStore;

    run_store_contract!(
        #[sqlx::test(migrator = "drew::store::MIGRATOR")] (pool: sqlx::PgPool) => PgRunStore::new(pool)
    );
}
