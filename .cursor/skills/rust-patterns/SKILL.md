---
name: rust-patterns
description: >-
  Router for idiomatic Rust: invalid states, ownership, eros error sets,
  traits, builders, iterators, concurrency, unsafe, module layout, and
  port-and-adapter testing, plus the shared done check. Use when writing,
  reviewing, or refactoring Rust.
---

# Rust patterns

The scope depends on the task:

- **Change** (a feature or fix): apply these skills to Rust you add or whose contracts you are already changing. Leave unrelated functions on their current types.
- **Audit or cleanup** (the task asks for one): apply them to all code the task names. An audit reports each violation with its rule; a cleanup fixes them. Existing code is in scope.

The public-API bar is the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/).

Read each skill below whose trigger matches the work, and apply its rules and its "Before finishing" list.

| Skill | Read it when |
| --- | --- |
| [`rust-invalid-states`](../rust-invalid-states/SKILL.md) | A domain type, enum, `bool` or `Option` flag, newtype, typestate, or parse boundary is added or changed |
| [`rust-ownership`](../rust-ownership/SKILL.md) | A signature chooses between borrowing and owning, or code clones or uses `Cow` |
| [`rust-errors`](../rust-errors/SKILL.md) | A function is fallible, or code adds error types, context, `unwrap`, or `expect` |
| [`rust-traits-generics`](../rust-traits-generics/SKILL.md) | Code defines a trait, a generic function, a `dyn` object, or a conversion |
| [`rust-builders`](../rust-builders/SKILL.md) | A struct gains many options, or a public type needs a constructor |
| [`rust-iterators`](../rust-iterators/SKILL.md) | Code loops over a collection or collects results |
| [`rust-concurrency`](../rust-concurrency/SKILL.md) | Code uses threads, channels, shared state, async, or a network client |
| [`rust-unsafe`](../rust-unsafe/SKILL.md) | Code uses `unsafe`, FFI, raw pointers, or `transmute` |
| [`rust-modules`](../rust-modules/SKILL.md) | Code adds a module or file, sets visibility, or writes doc comments |
| [`rust-port-adapter-testing`](../rust-port-adapter-testing/SKILL.md) | Tests cover code that calls a database, file system, HTTP service, or mock |

## Done check

The change is done when the repo's documented checks pass and every "Before finishing" item of each skill you read holds. The repo's `AGENTS.md`, contributing guide, or CI config is the source of truth for those checks. Without one, run:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

`cargo nextest run` skips doctests, so a repo on nextest also runs `cargo test --doc`. A lint override states the guideline it is departing from.
