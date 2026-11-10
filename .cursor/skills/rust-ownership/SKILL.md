---
name: rust-ownership
description: >-
  Rust ownership and borrowing in function signatures: references versus owned
  parameters, &str and AsRef, deliberate clones, and Cow. Use when writing Rust
  function parameters, removing clones, or fixing borrow-checker errors.
---

# Rust ownership and borrowing

Part of [`rust-patterns`](../rust-patterns/SKILL.md), which sets the scope and the done check.

Pass a reference when the function does not store or consume the value. Take ownership when it does. The caller decides when to clone.

```rust
fn process(data: &[u8]) -> usize {
    data.len()
}

fn store(data: Vec<u8>) -> Record {
    Record { payload: data }
}

fn process_bad(data: &Vec<u8>) -> usize {
    let cloned = data.clone();
    cloned.len()
}
```

Borrow `&str` or `impl AsRef<str>` when a `String` is not stored. A trait method called through `dyn` takes `&str` or `&[&str]`: `impl AsRef<str>` and other generic parameters make the trait not dyn-compatible. `Cow` is for a function that usually borrows and sometimes allocates:

```rust
use std::borrow::Cow;

fn normalize(input: &str) -> Cow<'_, str> {
    if input.contains(' ') {
        Cow::Owned(input.replace(' ', "_"))
    } else {
        Cow::Borrowed(input)
    }
}
```

## Before finishing

- Each clone and each `String` parameter has a reason in the code's shape.
