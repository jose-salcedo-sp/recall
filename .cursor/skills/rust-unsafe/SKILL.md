---
name: rust-unsafe
description: >-
  When Rust unsafe code is justified, and how to document it with Safety
  sections and SAFETY comments. Use when writing or reviewing unsafe blocks,
  FFI, raw pointers, or transmute.
---

# Rust unsafe

Part of [`rust-patterns`](../rust-patterns/SKILL.md), which sets the scope and the done check.

`unsafe` is for an FFI boundary or a measured hotspot whose invariant is proved. The function has a `# Safety` section. Each `unsafe` block has a `SAFETY` comment that states the invariant the compiler cannot check.

```rust
/// # Safety
///
/// `ptr` must be a valid, aligned pointer to an initialized `Widget`.
unsafe fn widget_from_raw<'a>(ptr: *const Widget) -> &'a Widget {
    // SAFETY: caller guarantees ptr is valid and aligned
    unsafe { &*ptr }
}
```

Borrow-checker trouble, convenience, a missing `SAFETY` comment, and `transmute` between unrelated types stay in safe code.
