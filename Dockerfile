# syntax=docker/dockerfile:1

# Keep in step with rust-toolchain.toml.
ARG RUST_VERSION=1.99.0

FROM rust:${RUST_VERSION}-slim-trixie AS build
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked --bin drew \
    && cp target/release/drew /usr/local/bin/drew

FROM debian:trixie-slim
# rustls-platform-verifier reads the system trust store.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin drew
COPY --from=build /usr/local/bin/drew /usr/local/bin/drew
COPY config/drew.json /etc/drew/drew.json
ENV DREW_CONFIG=/etc/drew/drew.json
# Numeric, so Kubernetes `runAsNonRoot` can verify it.
USER 10001
ENV DREW_BIND=0.0.0.0:8080
EXPOSE 8080
ENTRYPOINT ["drew"]
CMD ["serve"]
