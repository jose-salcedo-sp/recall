---
name: rust-traits-generics
description: >-
  Rust traits and generics: static versus dynamic dispatch, impl Trait, object
  safety, Send and Sync bounds, sealed traits, Read and Write parameters, and
  From and TryFrom conversions. Use when designing Rust traits, generic
  functions, trait objects, or type conversions.
---

# Rust traits and generics

Part of [`rust-patterns`](../rust-patterns/SKILL.md), which sets the scope and the done check.

Accept the most general input that stays clear. Return a concrete type or `impl Trait`.

```rust
fn read_all(reader: &mut impl Read) -> std::io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf)?;
    Ok(buf)
}

fn process<T: Display + Send + 'static>(item: T) -> String {
    format!("processed: {item}")
}
```

Static dispatch is the default: `impl Trait` in argument position, or a type parameter when the bounds are clearer in a `where` clause. `dyn Trait` is for a heterogeneous collection or a value selected at runtime. Make the trait object-safe when that use is real. Bounds include `Send` and `Sync` when the value crosses threads. Seal a trait that downstream crates must not implement.

Integration tests and doctests are downstream crates. A trait they implement stays unsealed: sealing it makes a `compile_fail` doctest that implements it fail on the seal (E0277) instead of on the error it exists to prove, and the doctest still passes. An error code (`compile_fail,E0599`) catches this only on nightly, since stable rustdoc ignores the code.

```rust
trait Handler: Send + Sync {
    fn handle(&self, request: &Request) -> Response;
}

struct Router {
    handlers: Vec<Box<dyn Handler>>,
}

fn fast_process<H: Handler>(handler: &H, request: &Request) -> Response {
    handler.handle(request)
}
```

Generic reader and writer parameters take `R: Read` and `W: Write` by value. Callers pass `&mut file` because `&mut R` implements `Read`.

Implement `From` and `TryFrom` for conversions. `Into` and `TryInto` come from the blanket impl. `Deref` is only for smart pointers.

## Before finishing

- Each `dyn` has a reason: a heterogeneous collection or a runtime-selected value.
- No sealed trait is implemented by an integration test or doctest.
