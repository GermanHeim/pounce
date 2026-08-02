# syntax=docker/dockerfile:1
#
# POUNCE — source build. Compiles the working tree you hand it, so this is
# the image for "the current commit": a PR branch, an unreleased fix, a
# bisect. For a released version use docker/Dockerfile.release, which pulls
# prebuilt wheels from PyPI in seconds instead of compiling for minutes.
#
# Build from the REPOSITORY ROOT (the context must include Cargo.toml,
# crates/, python/ and pyomo-pounce/ — see .dockerignore):
#
#   docker build -f docker/Dockerfile -t pounce:dev \
#     --build-arg POUNCE_BUILD_GIT="$(git rev-parse --short=8 HEAD)" .
#
# or just `make docker`, which fills in the SHA for you.
#
# The image carries both interfaces, because on a cluster you rarely know in
# advance which one a job will want:
#   * the `pounce` CLI (plus `pounce_sens` and `pounce_cblib`) on PATH
#   * `import pounce` — the Python API, with numpy/scipy
#   * `pyomo-pounce` — SolverFactory('pounce') from a Pyomo model
#
# NOT built with `--features ma57`: CoinHSL is license-restricted and cannot
# be redistributed in an image. The pure-Rust FERAL backend is the default
# and needs no external libraries.

# Any toolchain new enough for edition 2024 / resolver 3 (>= 1.85) works;
# pinned rather than floating so an image rebuilt months from now compiles
# the same way. CI builds on `stable`, so bumping this is safe and routine.
ARG RUST_VERSION=1.96
# abi3-py39 means the extension module works on any newer CPython, so the
# runtime Python version is a free choice; it is not tied to the builder's.
ARG PYTHON_VERSION=3.12
# Used for BOTH stages, and they must stay in step: the runtime's glibc has
# to be at least the builder's or the binaries it copies will not start.
# Nothing external forces the choice here — this image compiles everything it
# runs — but it matches docker/Dockerfile.release, which does not have that
# freedom (see the note there). One base, one story.
ARG DEBIAN_RELEASE=trixie

# ---------------------------------------------------------------------------
# Stage 1 — compile the CLI and build the pounce-solver wheel.
# ---------------------------------------------------------------------------
FROM rust:${RUST_VERSION}-${DEBIAN_RELEASE} AS builder

# Redeclared: ARGs before the first FROM are outside every build stage.
ARG DEBIAN_RELEASE

# Stamped into `pounce --about` so an image can name the commit it came from.
# .git is deliberately not in the build context (see .dockerignore), so the
# value has to be passed in; it reads "unknown" if you omit it, which is
# honest but unhelpful when someone hands you a container and asks what is in
# it. crates/pounce-cli/build.rs consumes this env var directly.
ARG POUNCE_BUILD_GIT=unknown

# Note on SOURCE_DATE_EPOCH: build.rs honors it for a reproducible build
# timestamp, but do NOT plumb it through as an ARG here. An ARG with no
# default exports as the empty string, and cargo-cyclonedx (pulled in by
# maturin) parses the variable unconditionally — an empty value panics the
# wheel build with "Invalid reproducible build timestamp: ParseIntError".
# Set it in the environment for a specific build if you need it.

ENV DEBIAN_FRONTEND=noninteractive

# maturin needs a Python interpreter and headers to build the extension.
# python3-venv gets us a writable environment to pip-install maturin into
# without fighting Debian's PEP 668 externally-managed marker.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/build-venv
ENV PATH="/opt/build-venv/bin:${PATH}"
RUN pip install --no-cache-dir "maturin>=1.4,<2.0"

# Must be set before the compile, not after: build.rs reads it at compile
# time and bakes the result into the binary with cargo:rustc-env.
ENV POUNCE_BUILD_GIT=${POUNCE_BUILD_GIT}

WORKDIR /src
COPY . .

# One RUN for the whole compile. The cache mounts make an incremental local
# rebuild cheap, but their contents do NOT survive into the next layer — so
# every artifact the runtime stage needs must be copied out of /src/target
# to /out before this command ends. Splitting this into several RUN steps
# would silently produce an image with no binaries in it.
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/src/target,sharing=locked \
    set -eux; \
    cargo build --release --locked \
        --bin pounce --bin pounce_sens --bin pounce_cblib; \
    mkdir -p /out/bin; \
    # Only the two auxiliary binaries. `pounce` itself is deliberately NOT
    # staged here: the wheel installs a console script of the same name into
    # the same directory, so pip would overwrite whatever we put there and
    # the copy would be a silent no-op. Letting the wheel own it also makes
    # this image byte-for-byte equivalent to the release image in how the
    # CLI resolves — the shim os.execv's straight into the bundled binary,
    # so signals and exit codes are unaffected.
    cp target/release/pounce_sens target/release/pounce_cblib /out/bin/; \
    # The wheel ships the CLI inside the package (python/pounce/_cli.py execs
    # it), and maturin only builds the extension module — nothing stages the
    # binary on its own. Same step `make python-cli-bin` performs locally.
    install -d python/pounce/bin; \
    install -m 0755 target/release/pounce python/pounce/bin/pounce; \
    # --compatibility linux tags the wheel linux_<arch> rather than running
    # auditwheel for a manylinux tag. We install it in the very next stage on
    # the same glibc, so a portable tag buys nothing and auditwheel's vendoring
    # would only add weight.
    cd python && maturin build --release --compatibility linux --out /out/wheels

# ---------------------------------------------------------------------------
# Stage 2 — runtime. No Rust toolchain, no build deps.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_RELEASE} AS runtime

ARG POUNCE_BUILD_GIT=unknown
ARG POUNCE_VERSION=dev

LABEL org.opencontainers.image.title="POUNCE" \
      org.opencontainers.image.description="Interior-point optimization solver (CLI + Python + Pyomo), built from source" \
      org.opencontainers.image.source="https://github.com/jkitchin/pounce" \
      org.opencontainers.image.documentation="https://jkitchin.github.io/pounce/" \
      org.opencontainers.image.licenses="EPL-2.0" \
      org.opencontainers.image.revision="${POUNCE_BUILD_GIT}" \
      org.opencontainers.image.version="${POUNCE_VERSION}"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /out/bin/ /usr/local/bin/
COPY --from=builder /out/wheels/ /tmp/wheels/
COPY --from=builder /src/pyomo-pounce/ /tmp/pyomo-pounce/
COPY --from=builder /src/LICENSE /usr/share/doc/pounce/LICENSE

# pyomo-pounce declares `pounce-solver>=X`, which pip would otherwise satisfy
# from PyPI — quietly replacing the wheel we just compiled with a released
# one. Installing both in a single resolve with the local wheel present lets
# it satisfy that requirement without reaching for the network.
RUN set -eux; \
    pip install --no-cache-dir /tmp/wheels/*.whl /tmp/pyomo-pounce; \
    rm -rf /tmp/wheels /tmp/pyomo-pounce

# Non-root by default. Apptainer/Singularity on a cluster runs as the calling
# user regardless, but under plain `docker run` this keeps output files in a
# bind-mounted /work from landing root-owned on the host.
RUN useradd --create-home --uid 1000 pounce \
    && mkdir -p /work \
    && chown pounce:pounce /work
USER pounce
WORKDIR /work

# Fail the build rather than ship a broken image. Covers all three surfaces:
# the CLI can solve (--problem needs no input file), the extension module
# imports, and the Pyomo plugin is registered and finds the executable.
RUN set -eux; \
    pounce --version; \
    pounce --problem rosenbrock --no-sol >/dev/null; \
    python -c "import pounce; print('pounce', pounce.__version__)"; \
    python -c "import pyomo.environ as pyo; assert pyo.SolverFactory('pounce').available(exception_flag=False)"

ENTRYPOINT ["pounce"]
CMD ["--help"]
