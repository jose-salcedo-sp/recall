# Multi-stage build for the Recall Rust service.
# Arch-agnostic: builds for the container runtime's native platform (arm64 locally, amd64 on RunPod).

# 1.94+ required: sqlx 0.9, pulled in by pgvector 0.4, sets that MSRV.
FROM rust:1.98 AS builder
WORKDIR /app

# Cache dependency compilation separately from application source.
COPY Cargo.toml Cargo.lock ./
RUN mkdir src \
    && echo 'fn main() {}' > src/main.rs \
    && cargo build --release \
    && rm -rf src

COPY src ./src
RUN touch src/main.rs && cargo build --release

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/recall /usr/local/bin/recall

EXPOSE 8000
ENTRYPOINT ["recall"]
