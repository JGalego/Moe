# A container for an engine whose whole point is having no runtime dependencies,
# so the final image has none either: one static-ish binary on a bare base.
#
#     docker build -t moe .
#     docker run --rm -v "$PWD/models:/models" moe info /models/mixtral
#     docker run --rm -p 8080:8080 -v moe-cache:/cache moe serve <repo> --host 0.0.0.0
#
# Weights are large and worth keeping across runs, so mount a volume at /cache;
# MOE_CACHE points the engine at it.

FROM rust:1.85-slim AS build
WORKDIR /src
# A C compiler and CA certificates, for the TLS stack the Hub client links.
RUN apt-get update \
 && apt-get install -y --no-install-recommends pkg-config libssl-dev ca-certificates \
 && rm -rf /var/lib/apt/lists/*
# Dependencies first, so editing the engine does not rebuild them.
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src \
 && echo 'fn main() {}' > src/main.rs \
 && echo '' > src/lib.rs \
 && cargo build --release --locked \
 && rm -rf src
COPY . .
# Touch the real sources so the stub build above does not satisfy the timestamps.
RUN touch src/main.rs src/lib.rs && cargo build --release --locked

FROM debian:bookworm-slim
# The engine reaches HTTPS for the Hub, so it needs the trust store — and nothing
# else. No Python, no BLAS, no CUDA.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/moe /usr/local/bin/moe
# Serving binds 127.0.0.1 by default, which is unreachable from outside a
# container; pass --host 0.0.0.0 deliberately rather than having it defaulted.
ENV MOE_CACHE=/cache
VOLUME /cache
EXPOSE 8080
ENTRYPOINT ["moe"]
CMD ["--help"]
