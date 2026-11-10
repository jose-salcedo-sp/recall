---
name: rust-modules
description: >-
  Rust module layout by domain, pub versus pub(crate) visibility, root
  re-exports, and what comments and doc comments say. Use when adding Rust
  modules or files, choosing item visibility, or writing Rust docs and comments.
---

# Rust modules and visibility

Part of [`rust-patterns`](../rust-patterns/SKILL.md), which sets the scope and the done check.

Organize by domain, not by type (`models/`, `errors/`, `utils/`). An item that exists for one caller lives next to that caller.

```text
my_app/
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── auth/
│   │   ├── mod.rs
│   │   ├── token.rs
│   │   └── middleware.rs
│   ├── orders/
│   │   ├── mod.rs
│   │   ├── model.rs
│   │   └── service.rs
│   └── db/
│       ├── mod.rs
│       └── pool.rs
├── tests/
├── benches/
└── Cargo.toml
```

`pub` is the crate's external API. Its consumers include the crate's own binaries, integration tests, benches, and doctests, so an item one of them uses is `pub`. Everything else is private or `pub(crate)`. Re-export that API from the crate root. Helpers stay private; tests in the same module can see them.

```rust
pub(crate) fn validate_input(input: &str) -> bool {
    !input.is_empty()
}

pub mod auth;
pub use auth::AuthMiddleware;
```

Comments and doc comments say why: the invariant, the failure mode, the safety requirement. A doc example is optional; write one where the signature does not show the usage. An example that calls a fallible function propagates with `?`, not `unwrap`.

## Before finishing

- New items are private unless an external consumer, including a test, bench, or doctest, uses them.
