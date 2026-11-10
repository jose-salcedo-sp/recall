---
name: rust-port-adapter-testing
description: >-
  Rust test pattern in ASD-STE100 Simplified Technical English: dependency
  injection, ports and adapters, one contract test suite per port, and
  in-memory stub adapters in place of mocks. Use when writing or reviewing
  Rust tests for code that calls a database, file system, HTTP service, or
  other external system, or when a test uses mockall or another mock library.
---

# Rust port and adapter testing

This skill is part of [`rust-patterns`](../rust-patterns/SKILL.md). That skill gives the scope and the done check.

This skill uses ASD-STE100 Simplified Technical English. Write new text for this skill in the same style:

- Write one instruction in each sentence.
- Use the active voice and the imperative for instructions.
- Keep an instruction to 20 words or fewer. Keep a description to 25 words or fewer.
- Use one term for one thing. This skill uses the terms in the table below.

| Term | Meaning |
|------|---------|
| Port | A trait that tells how the core code uses an external system. |
| Adapter | A type that implements a port for one external system. |
| Stub adapter | An adapter that keeps its data in memory. It does not connect to an external system. |
| Contract test | A test that is written for a port. Each adapter must pass it. |
| Failing adapter | An adapter that returns an error from each method. It does not run the contract tests. |

## Rules

1. Give each dependency to its user from outside. Do not construct a dependency inside the type that uses it.
2. Put the configuration for an external system in a struct. Read the environment in one place only, such as the CLI parser or one `from_env` function. Give an SDK its configuration explicitly, so that it does not read the environment again.
3. Write one port trait for each capability. One external system can supply more than one port.
4. Write one adapter for each backend.
5. Write the contract tests for the port. Do not write them for one adapter.
6. Run all contract tests on each adapter, and on the stub adapter too.
7. Use the stub adapter in unit tests. Use it in place of a mock.
8. When an adapter fails a contract test, change the adapter. Do not change the test to agree with one backend.
9. Use a failing adapter to test error paths. Write one for each port.

**WARNING:** A mock contains your guess about how a dependency behaves. If the guess is wrong, the test passes and the code fails in production.

**NOTE:** Small differences between adapters cause production bugs. Examples are list order and case-sensitive search. Write a contract test for each of these rules.

**NOTE:** Some ports have no fixed result, for example a language model. For these ports, the contract tests check shape and transport: count, order, dimensions, IDs, and error mapping. The stub adapter for such a port can return scripted content. The scripted content must obey the contract.

## Procedure

1. Find each external system that the code calls.
2. Move the configuration of each system into a config struct.
3. Change the constructors so that they receive the config or the connection.
4. Write a port trait. Use the domain types in its methods. Do not use the backend types.
5. Move the backend code into an adapter that implements the port.
6. Write the contract tests as generic functions on the port.
7. Write a macro that makes one test function for each contract test. For async ports, use Example 6.
8. Call the macro once for each adapter.
9. Write the stub adapter. Run the contract tests on it until all tests pass.
10. Replace each mock in the unit tests with the stub adapter.

The work is complete when each adapter and the stub adapter pass the full contract suite. Unit tests must not have mocks for the port.

## Examples

### Example 1: Inject the dependency

Incorrect. The store reads the environment and connects to the database. All errors from the constructor come from other parts of the code.

```rust
impl MySqlUserStore {
    pub fn new() -> Result<Self, StoreError> {
        let host = std::env::var("MYSQL_HOST")?;
        let password = std::env::var("MYSQL_PASSWORD")?;
        let pool = mysql::Pool::new(build_url(&host, &password))?;
        Ok(Self { pool })
    }
}
```

Correct. Each part does one task. The caller gives the connection to the store.

```rust
pub struct MySqlConfig {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub password: String,
}

impl MySqlConfig {
    pub fn from_env() -> Result<Self, ConfigError> {
        // All environment errors stay in this function.
    }
}

pub struct MySqlConnection { pool: mysql::Pool }

impl MySqlConnection {
    pub fn connect(config: &MySqlConfig) -> Result<Self, ConnectError> {
        // All connection errors stay in this function.
    }
}

pub struct MySqlUserStore { connection: MySqlConnection }

impl MySqlUserStore {
    pub fn new(connection: MySqlConnection) -> Self {
        Self { connection }
    }
}
```

### Example 2: Write the port

The port uses domain types only. The core code does not know which backend it uses.

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct User {
    pub username: String,
    pub email: String,
}

#[derive(Debug, PartialEq, Eq)]
pub enum StoreError {
    DuplicateUsername,
    Backend(String),
}

pub trait UserStore {
    fn add(&mut self, user: User) -> Result<(), StoreError>;
    fn get_by_username(&self, username: &str) -> Result<Option<User>, StoreError>;
}
```

### Example 3: Write the stub adapter

The stub adapter keeps users in a `Vec`. It must obey the same rules as the other adapters. Here, a username must be unique.

```rust
#[derive(Default)]
pub struct InMemoryUserStore {
    users: Vec<User>,
}

impl UserStore for InMemoryUserStore {
    fn add(&mut self, user: User) -> Result<(), StoreError> {
        if self.users.iter().any(|existing| existing.username == user.username) {
            return Err(StoreError::DuplicateUsername);
        }
        self.users.push(user);
        Ok(())
    }

    fn get_by_username(&self, username: &str) -> Result<Option<User>, StoreError> {
        Ok(self.users.iter().find(|user| user.username == username).cloned())
    }
}
```

### Example 4: Write the contract tests

Write each contract test one time, as a generic function on the port.

```rust
// src/user_store/contract.rs
use super::{StoreError, User, UserStore};

fn alice() -> User {
    User { username: "alice".into(), email: "alice@example.com".into() }
}

pub fn returns_user_by_username(store: &mut impl UserStore) {
    store.add(alice()).unwrap();

    let found = store.get_by_username("alice").unwrap();

    assert_eq!(found, Some(alice()));
}

pub fn returns_none_for_unknown_username(store: &mut impl UserStore) {
    let found = store.get_by_username("nobody").unwrap();

    assert_eq!(found, None);
}

pub fn rejects_duplicate_username(store: &mut impl UserStore) {
    store.add(alice()).unwrap();

    let result = store.add(alice());

    assert_eq!(result, Err(StoreError::DuplicateUsername));
}

#[macro_export]
macro_rules! user_store_contract_tests {
    ($make_store:expr) => {
        #[test]
        fn returns_user_by_username() {
            $crate::user_store::contract::returns_user_by_username(&mut $make_store);
        }

        #[test]
        fn returns_none_for_unknown_username() {
            $crate::user_store::contract::returns_none_for_unknown_username(&mut $make_store);
        }

        #[test]
        fn rejects_duplicate_username() {
            $crate::user_store::contract::rejects_duplicate_username(&mut $make_store);
        }
    };
}
```

When you add a contract test, add it to the macro. Then each adapter runs it.

### Example 5: Run the contract tests on each adapter

Call the macro once for each adapter. Each call makes a new store for each test, so the tests stay isolated.

```rust
#[cfg(test)]
mod in_memory_contract {
    use crate::user_store::InMemoryUserStore;
    crate::user_store_contract_tests!(InMemoryUserStore::default());
}

#[cfg(test)]
mod json_file_contract {
    use crate::user_store::JsonFileUserStore;
    crate::user_store_contract_tests!(JsonFileUserStore::new(tempfile::tempdir().unwrap()));
}

#[cfg(all(test, feature = "mysql-tests"))]
mod mysql_contract {
    use crate::user_store::{MySqlConfig, MySqlConnection, MySqlUserStore};
    crate::user_store_contract_tests!(MySqlUserStore::new(
        MySqlConnection::connect(&MySqlConfig::from_env().unwrap()).unwrap()
    ));
}
```

**NOTE:** When a server can start locally, run its contract tests in the default test run. An example is a database in a container. Use a Cargo feature, as above, only when the server cannot run where the tests run. Test an HTTP adapter against a local HTTP test server, such as `wiremock`. Copy its responses from the service documentation or from recorded traffic.

### Example 6: Test an async port

Many ports are async and shared, for example `Arc<dyn UserStore>`. Write each contract test as an async function on `&dyn UserStore`. A stub adapter for a `&self` port keeps its data in a `Mutex`.

Some test harnesses give a resource to the test as an argument. For example, `#[sqlx::test]` gives each test a new database pool. Let the macro caller give the test attribute, the arguments, and the expression that makes the adapter.

```rust
#[async_trait::async_trait]
pub trait UserStore: Send + Sync {
    async fn add(&self, user: User) -> Result<(), StoreError>;
}

pub async fn rejects_duplicate_username(store: &dyn UserStore) {
    store.add(alice()).await.unwrap();

    let result = store.add(alice()).await;

    assert_eq!(result, Err(StoreError::DuplicateUsername));
}

#[macro_export]
macro_rules! user_store_contract_tests {
    (#[$attr:meta] ($($arg:ident: $ty:ty),*) => $make_store:expr) => {
        #[$attr]
        async fn rejects_duplicate_username($($arg: $ty),*) {
            let store = $make_store;
            $crate::user_store::contract::rejects_duplicate_username(&store).await;
        }
    };
}

#[cfg(test)]
mod in_memory_contract {
    use crate::user_store::InMemoryUserStore;
    crate::user_store_contract_tests!(#[tokio::test] () => InMemoryUserStore::default());
}

#[cfg(test)]
mod postgres_contract {
    use crate::user_store::PgUserStore;
    crate::user_store_contract_tests!(#[sqlx::test] (pool: sqlx::PgPool) => PgUserStore::new(pool));
}
```

### Example 7: Use the stub adapter in a unit test

The service receives its store from outside. The unit test gives it the stub adapter. The test does not use a mock.

```rust
pub struct SignupService<TStore: UserStore> {
    store: TStore,
}

impl<TStore: UserStore> SignupService<TStore> {
    pub fn new(store: TStore) -> Self {
        Self { store }
    }

    pub fn register(&mut self, username: &str, email: &str) -> Result<(), SignupError> {
        let user = User { username: username.into(), email: email.into() };
        self.store.add(user).map_err(SignupError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_second_signup_with_same_username() {
        let mut service = SignupService::new(InMemoryUserStore::default());
        service.register("alice", "alice@example.com").unwrap();

        let result = service.register("alice", "other@example.com");

        assert_eq!(result, Err(SignupError::UsernameTaken));
    }
}
```

The test is correct only if the stub adapter is correct. The contract tests in Example 5 prove that.

### Example 8: Replace a mock

Incorrect. The mock tells what the store returns. Nothing proves that a real store returns the same value.

```rust
let mut store = MockUserStore::new();
store.expect_add().returning(|_| Err(StoreError::DuplicateUsername));
let mut service = SignupService::new(store);
```

Correct. The stub adapter makes the duplicate error from its own rules. The contract tests prove those rules.

```rust
let mut service = SignupService::new(InMemoryUserStore::default());
service.register("alice", "alice@example.com").unwrap();
```

## Review checklist

- [ ] No constructor reads the environment or opens a connection for a dependency.
- [ ] Each capability has one port trait.
- [ ] The port methods use domain types only.
- [ ] Each adapter, and the stub adapter, runs the full contract macro.
- [ ] The contract suite has a test for each rule: uniqueness, order, case, and "not found".
- [ ] Unit tests use the stub adapter. They do not use mocks for the port.
- [ ] Error-path tests use a failing adapter.

## Production use of stub adapters

Stub adapters are also useful outside tests. Examples are a cache in front of a slow adapter, a fast backend for batch jobs, and a test double that you publish for library users. Keep the stub adapter in the contract suite when you use it in production.
