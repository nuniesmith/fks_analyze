# Multi-stage Dockerfile extending shared Rust template

# Build stage (using official Rust image)
FROM rust:1.82-slim AS build

WORKDIR /src

# Copy Rust project files
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# Build the application
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked && \
    cp target/release/fks_analyze /usr/local/bin/app

ARG SERVICE_PORT=4802

# Runtime stage - debian slim (needs shell & tools for dynamic analysis scripts)
FROM debian:bookworm-slim AS final
ARG SERVICE_PORT=4802
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata bash curl git ripgrep file \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy CA certificates and timezone data
COPY --from=build /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=build /usr/share/zoneinfo /usr/share/zoneinfo

# Copy the built binary
COPY --from=build /usr/local/bin/app /usr/local/bin/app

# Set service-specific environment variables
ENV SERVICE_NAME=fks_analyze \
    SERVICE_TYPE=analyze \
    SERVICE_PORT=${SERVICE_PORT} \
    RUST_LOG=info \
    FKS_ANALYZE_ROOT=/fks_root \
    FKS_ANALYZE_INDEX=/app/index

EXPOSE ${SERVICE_PORT}

# Create non-root user
RUN useradd -u 1089 -m appuser && chown -R appuser /app
USER appuser

ENTRYPOINT ["/usr/local/bin/app"]
