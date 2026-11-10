---
name: rust-builders
description: >-
  Rust constructors and builders: when a value needs a builder, constructor
  shape, private fields, and Debug and Default on public types. Use when a Rust
  struct gains many options or a new public type needs a constructor.
---

# Rust builders

Part of [`rust-patterns`](../rust-patterns/SKILL.md), which sets the scope and the done check.

A builder is for a value with many independent options. A handful of fields is a constructor or `Default` plus struct update. Constructors are inherent associated functions, and struct fields stay private. Implement `Debug` on public types, and `Default` when a meaningful default exists.

```rust
struct ServerConfig {
    host: String,
    port: u16,
    max_connections: usize,
}

impl ServerConfig {
    fn builder(host: impl Into<String>, port: u16) -> ServerConfigBuilder {
        ServerConfigBuilder { host: host.into(), port, max_connections: 100 }
    }
}

struct ServerConfigBuilder {
    host: String,
    port: u16,
    max_connections: usize,
}

impl ServerConfigBuilder {
    fn max_connections(mut self, n: usize) -> Self {
        self.max_connections = n;
        self
    }

    fn build(self) -> ServerConfig {
        ServerConfig { host: self.host, port: self.port, max_connections: self.max_connections }
    }
}
```
