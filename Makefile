# POUNCE — Makefile wrapper around cargo for build, test, and install.
#
# Usage:
#   make                  # release build of the workspace
#   make build            # release build (alias)
#   make debug            # debug build
#   make test             # run all tests
#   make check            # cargo check
#   make clippy           # lint with clippy (treats warnings as errors)
#   make fmt              # rustfmt the workspace
#   make doc              # build rustdoc
#   make install          # install pounce CLI + cinterface cdylib under $(PREFIX)
#   make uninstall        # remove installed artifacts
#   make clean            # cargo clean
#
# Default install prefix is $(HOME)/.local — a user-owned directory
# that needs no sudo. Make sure $(HOME)/.local/bin is on your PATH
# (and $(HOME)/.local/lib on DYLD_LIBRARY_PATH / LD_LIBRARY_PATH if
# you intend to link against libpounce_cinterface from outside cargo).
#
# Override for a system-wide install (requires sudo):
#   sudo make install PREFIX=/usr/local
#
# Or pick any other user-owned directory:
#   make install PREFIX=$$HOME/opt/pounce
#
# Pass extra flags through to cargo:
#   make build CARGO_FLAGS="--features feral"

CARGO       ?= cargo
PREFIX      ?= $(HOME)/.local
BINDIR      ?= $(PREFIX)/bin
LIBDIR      ?= $(PREFIX)/lib
INCLUDEDIR  ?= $(PREFIX)/include
PROFILE     ?= release
CARGO_FLAGS ?=

TARGET_DIR    := target/$(PROFILE)
CLI_BIN       := $(TARGET_DIR)/pounce
CDYLIB_NAME   := libpounce_cinterface
UNAME_S       := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  CDYLIB_EXT := dylib
else ifeq ($(UNAME_S),Linux)
  CDYLIB_EXT := so
else
  CDYLIB_EXT := dll
endif
CDYLIB        := $(TARGET_DIR)/$(CDYLIB_NAME).$(CDYLIB_EXT)

ifeq ($(PROFILE),release)
  CARGO_PROFILE_FLAG := --release
else
  CARGO_PROFILE_FLAG :=
endif

.PHONY: all build debug test check clippy fmt fmt-check doc install uninstall clean help

all: build

build:
	$(CARGO) build --workspace $(CARGO_PROFILE_FLAG) $(CARGO_FLAGS)

debug:
	$(MAKE) build PROFILE=debug

test:
	$(CARGO) test --workspace $(CARGO_PROFILE_FLAG) $(CARGO_FLAGS)

check:
	$(CARGO) check --workspace $(CARGO_FLAGS)

clippy:
	$(CARGO) clippy --workspace --all-targets $(CARGO_FLAGS) -- -D warnings

fmt:
	$(CARGO) fmt --all

fmt-check:
	$(CARGO) fmt --all -- --check

doc:
	$(CARGO) doc --workspace --no-deps $(CARGO_PROFILE_FLAG)

install: build
	@echo "Installing pounce into $(PREFIX)"
	install -d "$(DESTDIR)$(BINDIR)" "$(DESTDIR)$(LIBDIR)"
	install -m 0755 "$(CLI_BIN)" "$(DESTDIR)$(BINDIR)/pounce"
	install -m 0644 "$(CDYLIB)" "$(DESTDIR)$(LIBDIR)/$(CDYLIB_NAME).$(CDYLIB_EXT)"

uninstall:
	@echo "Removing pounce from $(PREFIX)"
	rm -f "$(DESTDIR)$(BINDIR)/pounce"
	rm -f "$(DESTDIR)$(LIBDIR)/$(CDYLIB_NAME).$(CDYLIB_EXT)"

clean:
	$(CARGO) clean

help:
	@sed -n 's/^# \{0,1\}//p' Makefile | sed -n '1,30p'
