---
name: rust-iterators
description: >-
  Rust iterator chains versus for loops, and collect into Vec, HashMap, String,
  or Result. Use when writing Rust loops, filter and map chains, or collecting
  results.
---

# Rust iterators

Part of [`rust-patterns`](../rust-patterns/SKILL.md), which sets the scope and the done check.

Prefer an iterator chain when the work is a transformation: filter, map, collect. Keep a `for` loop when the body breaks early, mutates through several steps, or the chain would hide the effect.

```rust
let active_emails: Vec<String> = users.iter().filter(|u| u.is_active).map(|u| u.email.clone()).collect();
```

`collect` selects the container. Collecting `Result`s stops on the first error.

```rust
let names: Vec<_> = items.iter().map(|i| &i.name).collect();
let lookup: HashMap<_, _> = items.iter().map(|i| (i.id, i)).collect();
let combined: String = parts.iter().copied().collect();
let parsed: Result<Vec<i32>, _> = strings.iter().map(|s| s.parse()).collect();
```
