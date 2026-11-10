---
name: rust-errors
description: >-
  Rust error handling with eros error sets: typed error unions, union, widen,
  recover, #[context], thiserror domain errors, error boundaries, and the unwrap
  and expect policy. Use when writing fallible Rust functions, Result types,
  error enums, error context, or error reporting.
---

# Rust errors

Part of [`rust-patterns`](../rust-patterns/SKILL.md), which sets the scope and the done check.

Error composition follows the [error-set article](https://mcmah309.github.io/posts/the-missing-piece-in-rust-error-handling/): possible errors compose as a set, and handling one removes it from the set.

`eros` is that set. Add it when the first new fallible function needs it. Enable its `anyhow` feature only when the surrounding code returns `anyhow::Error`.

The same rules apply in libraries and applications. `eros::Result<T, (A, B)>` is `Result<T, ErrorUnion<(A, B)>>`: the tuple is the set of possibilities, and the value holds one of them. The caller decides between the two forms:

- **Some caller branches on an error**: return a typed set. It names every concrete error the function can return, including ones no caller branches on, such as a database error that rides along beside the one the caller `recover`s. A callee whose set exists only to be widened into such a set may be typed too, even as a singleton `(E,)`.
- **Every caller only propagates or reports**: return `eros::Result<T>`, which uses `AnyError`. Context still travels with an erased error.

A propagated error that is missing from the set does not compile. Add the type, or `recover` it. Erasing to `AnyError` is for callers that do not branch, not a way to silence the check.

```rust
use eros::{IntoUnion, ReshapeUnion, context};
use std::{io, num::ParseIntError};

type PortErrors = (io::Error, ParseIntError);

#[context("load server port from {}", path)]
fn load_port(path: &str) -> eros::Result<u16, PortErrors> {
    let contents = std::fs::read_to_string(path).union()?;
    contents.trim().parse().union()
}

fn load_host(path: &str) -> eros::Result<std::net::IpAddr, (io::Error, std::net::AddrParseError)> {
    let contents = std::fs::read_to_string(path).union()?;
    contents.trim().parse().union()
}

fn bind_server() -> eros::Result<std::net::TcpListener, (io::Error, std::net::AddrParseError, ParseIntError)> {
    let host = load_host("config/host.txt").widen()?;
    let port = load_port("config/port.txt").widen()?;
    std::net::TcpListener::bind((host, port)).union()
}

fn port_or_default(path: &str) -> eros::Result<u16, (ParseIntError,)> {
    load_port(path).recover::<io::Error, _>(|error| {
        eprintln!("{error:#?}; using port 8080");
        8080
    })
}
```

`.union()` wraps one concrete error into the set named by the signature. `.widen()` expands a smaller set into a larger one; a type that appears twice is listed once. `.recover::<E, _>` handles `E` and drops it from the set. `.recover::<(A, B), _>` handles a group. The handler's value is the success. `.try_recover` is the fallible form: the fallback's errors join whatever the handler did not consume. When the set is empty, `.into_value()` yields the success and does not compile while any error type remains.

A domain failure with its own payload is a unit or tuple struct that implements `Error + Display + Debug + Send + Sync`. `thiserror` is for that impl. The struct is then one member of the set. `Display` is lowercase and has no trailing punctuation. `eros::bail!` and `eros::error!` are the untyped form, for `AnyError`, when there is no structured variant to match.

```rust
#[derive(Debug, thiserror::Error)]
#[error("port must be nonzero, got {input:?}")]
struct ZeroPort { input: String }

fn parse_port(input: &str) -> eros::Result<u16, (ParseIntError, ZeroPort)> {
    let port = input.parse::<u16>().union()?;
    if port == 0 {
        return Err(ZeroPort { input: input.into() }).union();
    }
    Ok(port)
}
```

`Result<T, ()>` is not an error type. Use a unit struct, or `recover` until `.into_value()` applies.

## Context

The function owns its context. `#[context("...")]` names its operation and the inputs that explain a failure, on every error it returns. A call site adds `.context` or `.with_context` only for a fact the callee cannot see. The original error stays the headline. The trail is innermost first.

| Format | Shows |
| --- | --- |
| `{error:#?}` | Root, sources, and context |
| `{error:?}` | Those, plus locations and the backtrace |
| `{error}` | Root and `source()` chain |
| `{error:#}` | Root only |

Report a failure once, where the program handles or logs it. A recovered error is not reported again upstream.

## Boundaries

`.union()` accepts an ordinary `Result<T, E>`. `?` turns a typed set into `AnyError`. `ErrorUnion::from_anyhow` accepts `anyhow::Error` when that feature is on. A signature you are not allowed to change, such as a trait impl or a published type, converts at that function and stays as it is.

A published public API whose callers should not depend on `eros` can convert the set into a stable crate error at that function. Internal and application code keep the set. A wrapper enum whose variants are the errors of the functions it called, or one crate-wide enum that lists failures the function cannot return, is the pattern this set replaces.

`unwrap` and `expect` are for a bug invariant, with the invariant written in the `expect` message, and for tests. Doc examples propagate with `?`. A fallible read, parse, lock, or I/O call returns a `Result`. Public docs state what the function returns, when it panics, and any safety requirement.

## `Option` combinators

Chain `Option` with `map`, `and_then`, and `filter` when each step is the same shape. Match when an arm has its own logic.

```rust
fn find_user_email(users: &[User], id: u64) -> Option<String> {
    users.iter().find(|u| u.id == id).map(|u| u.email.clone())
}
```

## Before finishing

- Each new or touched fallible function is an `eros` result with `#[context]` on the function: a typed set naming every error it returns when some caller branches, `AnyError` otherwise.
- Domain errors implement `Error + Send + Sync`, and their `Display` is lowercase with no trailing punctuation.
- No new wrapper or crate-wide error enum was added to make composition compile.
- `unwrap` / `expect` remain only on a stated bug invariant or in tests.
