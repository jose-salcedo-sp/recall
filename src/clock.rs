//! Wall-clock time behind the [`Clock`] port and run identity behind the [`RunIds`] port, so traces, stored runs and
//! eval results are stamped by an injected source that tests can fix.
//!
//! Elapsed durations are still measured with [`std::time::Instant`], which is monotonic and is not the wall clock.

use chrono::{DateTime, Utc};

use crate::domain::RunId;

/// The current UTC time.
pub trait Clock: Send + Sync {
    /// Never earlier than a previous call on the same clock.
    fn now(&self) -> DateTime<Utc>;
}

/// The operating system's clock.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }
}

/// A fresh [`RunId`] per run.
pub trait RunIds: Send + Sync {
    /// Unique among the ids this source returns, and greater than every earlier one, so ids sort by creation.
    fn next(&self) -> RunId;
}

/// UUIDv7s: time-ordered, with random bits from the operating system.
#[derive(Clone, Copy, Debug, Default)]
pub struct UuidV7;

impl RunIds for UuidV7 {
    fn next(&self) -> RunId {
        RunId::new()
    }
}

/// A [`Clock`] that only moves when told to, and [`RunIds`] that count up from 1.
#[cfg(any(test, feature = "testing"))]
pub mod fakes {
    use std::sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    };

    use chrono::{DateTime, Duration, TimeZone, Utc};
    use uuid::Uuid;

    use super::{Clock, RunIds};
    use crate::domain::RunId;

    /// Stays at the time it was set to.
    #[derive(Debug)]
    pub struct FixedClock(Mutex<DateTime<Utc>>);

    impl FixedClock {
        /// Reads `at` until advanced.
        pub fn at(at: DateTime<Utc>) -> Self {
            Self(Mutex::new(at))
        }

        /// Moves the clock forward by `by`.
        pub fn advance(&self, by: Duration) {
            *self
                .0
                .lock()
                .expect("lock poisoned: a test thread panicked while holding it") += by;
        }
    }

    /// 2026-01-01T00:00:00Z.
    impl Default for FixedClock {
        fn default() -> Self {
            Self::at(
                Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0)
                    .single()
                    .expect("a valid date"),
            )
        }
    }

    impl Clock for FixedClock {
        fn now(&self) -> DateTime<Utc> {
            *self
                .0
                .lock()
                .expect("lock poisoned: a test thread panicked while holding it")
        }
    }

    /// `00000000-0000-0000-0000-000000000001`, then `…02`, and so on.
    #[derive(Debug, Default)]
    pub struct SequentialIds(AtomicU64);

    impl RunIds for SequentialIds {
        fn next(&self) -> RunId {
            let n = self.0.fetch_add(1, Ordering::Relaxed) + 1;
            RunId::from(Uuid::from_u128(u128::from(n)))
        }
    }
}

/// The rules every [`Clock`] and [`RunIds`] keeps; `clock_contract!` and `run_ids_contract!` turn them into tests.
#[cfg(any(test, feature = "testing"))]
pub mod contract {
    use std::collections::HashSet;

    use super::{Clock, RunIds};

    /// Reads in order never go backwards.
    pub fn time_never_goes_backwards(clock: &dyn Clock) {
        let mut last = clock.now();
        for _ in 0..1000 {
            let now = clock.now();
            assert!(now >= last, "{now} after {last}");
            last = now;
        }
    }

    /// Ids are unique and increasing, also when taken from several threads at once.
    pub fn ids_are_unique_and_increasing(ids: &(dyn RunIds + Sync)) {
        let mut last = ids.next();
        for _ in 0..1000 {
            let next = ids.next();
            assert!(next.as_uuid() > last.as_uuid(), "{next} after {last}");
            last = next;
        }
        let taken: Vec<_> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..4)
                .map(|_| scope.spawn(|| (0..250).map(|_| ids.next()).collect::<Vec<_>>()))
                .collect();
            workers
                .into_iter()
                .flat_map(|worker| worker.join().expect("an id thread panicked"))
                .collect()
        });
        let unique: HashSet<_> = taken.iter().collect();
        assert_eq!(unique.len(), taken.len());
    }

    /// One test per [`Clock`] check, for the clock `$make` builds.
    #[macro_export]
    macro_rules! clock_contract {
        ($make:expr) => {
            #[test]
            fn time_never_goes_backwards() {
                $crate::clock::contract::time_never_goes_backwards(&$make);
            }
        };
    }

    /// One test per [`RunIds`] check, for the source `$make` builds.
    #[macro_export]
    macro_rules! run_ids_contract {
        ($make:expr) => {
            #[test]
            fn ids_are_unique_and_increasing() {
                $crate::clock::contract::ids_are_unique_and_increasing(&$make);
            }
        };
    }
}

#[cfg(test)]
mod tests {
    mod system_clock {
        crate::clock_contract!(super::super::SystemClock);
    }

    mod fixed_clock {
        crate::clock_contract!(super::super::fakes::FixedClock::default());
    }

    mod uuid_v7 {
        crate::run_ids_contract!(super::super::UuidV7);
    }

    mod sequential_ids {
        crate::run_ids_contract!(super::super::fakes::SequentialIds::default());
    }

    #[test]
    fn a_fixed_clock_moves_only_when_advanced() {
        use super::{Clock, fakes::FixedClock};
        let clock = FixedClock::default();
        let start = clock.now();
        assert_eq!(clock.now(), start);
        clock.advance(chrono::Duration::seconds(5));
        assert_eq!(clock.now() - start, chrono::Duration::seconds(5));
    }
}
