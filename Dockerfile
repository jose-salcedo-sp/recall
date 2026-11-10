# syntax=docker/dockerfile:1

# Keep in step with rust-toolchain.toml.
ARG RUST_VERSION=1.99.0

FROM rust:${RUST_VERSION}-slim-trixie AS build
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked --bin recall \
    && cp target/release/recall /usr/local/bin/recall

FROM debian:trixie-slim
# rustls-platform-verifier reads the system trust store.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin recall
COPY --from=build /usr/local/bin/recall /usr/local/bin/recall
COPY config/recall.json /etc/recall/recall.json
ENV RECALL_CONFIG=/etc/recall/recall.json
COPY config/access.json /etc/recall/access.json
ENV RECALL_ACCESS=/etc/recall/access.json
# Numeric, so Kubernetes `runAsNonRoot` can verify it.
USER 10001
ENV RECALL_BIND=0.0.0.0:8080
EXPOSE 8080
ENTRYPOINT ["recall"]
CMD ["serve"]
