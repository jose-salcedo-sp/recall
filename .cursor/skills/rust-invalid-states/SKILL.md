---
name: rust-invalid-states
description: >-
  Rust type design that makes invalid states unrepresentable: enums for modes,
  newtypes, validated constructors, typestate, parsing at the boundary, and
  exhaustive matching. Use when adding or changing Rust domain types, structs
  with bool or Option flags, enums and match arms, or parsing external input.
---

# Rust invalid states

Part of [`rust-patterns`](../rust-patterns/SKILL.md), which sets the scope and the done check.

An invalid combination has no value the program can construct. Encode the constraint in the type, and check it once where unstructured input enters.

- A mode is an enum. A `bool`, or several `Option`s that only make sense together, is that enum.
- Two values of the same primitive that must not be swapped are newtypes.
- A value with an invariant (nonzero, non-empty, already parsed) is a type whose constructor is the check. Fields stay private. Downstream code trusts the type.
- A multi-step API whose steps are valid only in order is typestate: each state is its own type, and the transition consumes one state and returns the next.
- External input stays a string, number, or document until that boundary. `TryFrom` or `FromStr` is the conversion. A check stays at runtime only for a fact the type cannot see, such as a clock, I/O, or another process.

Add a type when it removes an invalid state or a repeated check. An ordinary string or config value stays ordinary.

## Enums and pattern matching

Match domain enums exhaustively. A wildcard arm is one real remainder, and it says why. Adding a variant should force every business match to be updated.

```rust
enum ConnectionState {
    Disconnected,
    Connecting { attempt: u32 },
    Connected { session_id: String },
    Failed { reason: String, retries: u32 },
}

fn handle(state: &ConnectionState) {
    match state {
        ConnectionState::Disconnected => connect(),
        ConnectionState::Connecting { attempt } if *attempt > 3 => abort(),
        ConnectionState::Connecting { .. } => wait(),
        ConnectionState::Connected { session_id } => use_session(session_id),
        ConnectionState::Failed { retries, .. } if *retries < 5 => retry(),
        ConnectionState::Failed { reason, .. } => log_failure(reason),
    }
}
```

```rust
match command {
    Command::Start => start_service(),
    Command::Stop => stop_service(),
    Command::Restart => restart_service(),
}
```

Parse unstructured input into these types at the boundary. A constructor that can fail returns `Result`. A `bool` or `Option` that is really a mode becomes its own enum.

## Newtypes

A newtype is for two parameters of the same primitive that must not be swapped, or for a wrapper that is the invariant. Implement `From`, `AsRef`, and the common traits that apply (`Copy`, `Clone`, `Eq`, `Hash`, `Debug`, `Display`).

```rust
struct UserId(u64);
struct OrderId(u64);

fn get_order(user: UserId, order: OrderId) -> eros::Result<Order> {
    todo!()
}
```

## Before finishing

- New or touched domain values make invalid states unrepresentable: an enum for a mode, a newtype when two primitives could be swapped, a private constructor when the value has an invariant, typestate when order is the invariant.
- Unstructured input is parsed at the boundary.
- Domain enums are matched exhaustively.
