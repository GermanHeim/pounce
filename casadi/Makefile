# Build the POUNCE CasADi `Nlpsol` plugin.
#
#   make                 build libcasadi_nlpsol_pounce.so
#   make test            build, then run the parity checks against ipopt
#   make examples        run every script in examples/
#   make compat          compile the convexify name shim against every
#                        CasADi spelling (needs no CasADi at all)
#   make clean
#
# Two inputs are needed and both are auto-detected where possible:
#
#   CASADI_LIB   directory holding libcasadi.so — taken from the installed
#                casadi Python package.
#   CASADI_SRC   a CasADi *source* tree at the SAME version, for the
#                internal headers (nlpsol_impl.hpp, plugin_interface.hpp,
#                …). The pip wheel ships only public headers, so this has
#                to be fetched:
#
#                    make fetch-src        # clones the matching tag
#                    make CASADI_SRC=/path/to/casadi
#
# The ABI flags below are not stylistic. CasADi's plugin loader performs
# no version handshake — a plugin built against a different CasADi, a
# different libstdc++ ABI, or a different -D set loads and then
# misbehaves. `abi-flags` reads them back from the installed CasADi so a
# mismatch shows up as a link error, not as a mystery.

PYTHON      ?= python3
CASADI_LIB  ?= $(shell $(PYTHON) -c "import casadi; print(casadi.GlobalOptions.getCasadiPath())" 2>/dev/null)
CASADI_INC  ?= $(shell $(PYTHON) -c "import casadi; print(casadi.GlobalOptions.getCasadiIncludePath())" 2>/dev/null)
CASADI_VER  ?= $(shell $(PYTHON) -c "import casadi; print(casadi.__version__)" 2>/dev/null)
CASADI_SRC  ?= $(CURDIR)/casadi-src

# Repo-relative default for POUNCE's C library and header.
POUNCE_ROOT ?= $(CURDIR)/..
POUNCE_INC  ?= $(POUNCE_ROOT)/crates/pounce-cinterface/include
POUNCE_LIB  ?= $(POUNCE_ROOT)/target/release

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  SO_EXT := dylib
  # `@loader_path` covers the layout the wheel ships: plugin and solver
  # library side by side in one directory, wherever pip put it.
  RPATH   = -Wl,-rpath,$(CASADI_LIB) -Wl,-rpath,$(POUNCE_LIB) \
            -Wl,-rpath,@loader_path
  # Rust stamps a cdylib's LC_ID_DYLIB with its absolute build path, so the
  # plugin links against `/abs/.../target/release/deps/libpounce_cinterface.dylib`
  # and BOTH -rpath entries above are inert for it: dyld resolves an absolute
  # install name directly and never consults the rpath list. Left alone the
  # plugin loads only while that exact build tree survives at that exact path
  # -- and when it does not, CasADi's loader swallows the dlopen error and
  # reports `Plugin 'pounce' is not found`, which sends you looking in the
  # wrong place entirely. Rewrite the reference to @rpath so the rpath list
  # above is what decides.
  #
  # Linux needs none of this: the reference there is the soname, which the
  # rpath already resolves.
  FIXUP = install_name_tool -change \
            "$$(otool -D $(POUNCE_LIB)/libpounce_cinterface.dylib | tail -1)" \
            @rpath/libpounce_cinterface.dylib $(PLUGIN)
else
  SO_EXT := so
  RPATH   = -Wl,-rpath,$(CASADI_LIB) -Wl,-rpath,$(POUNCE_LIB)
  FIXUP = :
endif

PLUGIN := libcasadi_nlpsol_pounce.$(SO_EXT)

# gh#667 regression driver. A C++ host, not a Python one: the Python
# bindings leave `Logger::flush` at its default, so the same check driven
# from test_parity.py would measure CasADi's buffering rather than the
# plugin's flushing. See the comment at the top of the source.
TEARTEST := test_output_interleaving

# Version macros CasADi's headers expect, derived from the installed
# CasADi rather than hard-coded.
VER_MAJOR := $(word 1,$(subst ., ,$(CASADI_VER)))
VER_MINOR := $(word 2,$(subst ., ,$(CASADI_VER)))
VER_PATCH := $(word 3,$(subst ., ,$(CASADI_VER)))

DEFS := -DWITH_DL -DWITH_DEEPBIND -DUSE_CXX11 -DHAVE_MKSTEMPS \
        -DCASADI_WITH_THREAD -DCASADI_WITH_THREADSAFE_SYMBOLICS \
        -DWITH_DEPRECATED_FEATURES \
        -DCASADI_MAJOR_VERSION=$(VER_MAJOR) \
        -DCASADI_MINOR_VERSION=$(VER_MINOR) \
        -DCASADI_PATCH_VERSION=$(VER_PATCH) \
        -DCASADI_VERSION=$(VER_MAJOR)$(VER_MINOR) \
        -DCASADI_IS_RELEASE=1

# Which libstdc++ string ABI the CasADi being built against uses. This is
# NOT a property of the CasADi version, and assuming it was is what broke
# every PR for a day (gh#782): casadi 3.8.0 added a `manylinux_2_28` wheel
# alongside the `manylinux2014` one, and pip picks between them on the
# machine's glibc. The 2_28 build of 3.8.0 exports 855 `casadi::` symbols
# spelled with `__cxx11` and none with the old `Ss`; the manylinux2014
# build of THE SAME VERSION is the other way round. So the answer changes
# with the runner, not with the release, and it has to be measured.
#
# Measure it on casadi's own exported symbols. A bare `__cxx11` grep is
# wrong: libstdc++'s transactional clones of `std::logic_error` carry that
# spelling in every build, old ABI included (213 of them in the wheels
# checked), so the naive probe answers 1 for everything.
#
# macOS is libc++, where the macro is inert; Windows has no libstdc++
# either. Both keep the historical 0, and no probe runs.
#
# With no `nm` the probe yields 0, which is the value this file assumed
# unconditionally until now — a wrong answer there is no more wrong than
# before, and `make abi-flags` prints what was decided.
ifeq ($(UNAME_S),Linux)
  CASADI_SO   ?= $(firstword $(wildcard $(CASADI_LIB)/libcasadi.so $(CASADI_LIB)/libcasadi.so.*))
  CXX11_ABI   ?= $(shell nm -D --defined-only "$(CASADI_SO)" 2>/dev/null \
                   | grep -qE '_ZN6casadi.*__cxx11' && echo 1 || echo 0)
else
  CXX11_ABI ?= 0
endif
DEFS      += -D_GLIBCXX_USE_CXX11_ABI=$(CXX11_ABI)

CXXFLAGS ?= -O2 -std=c++17 -fPIC -Wall
INCLUDES  = -I$(CASADI_SRC) -I$(CASADI_INC) -I$(POUNCE_INC)
LDFLAGS   = -L$(CASADI_LIB) -lcasadi -L$(POUNCE_LIB) -lpounce_cinterface $(RPATH)

.PHONY: all check-env fetch-src abi-flags install uninstall test examples compat clean

all: check-env $(PLUGIN)

check-env:
ifeq ($(strip $(CASADI_LIB)),)
	$(error CasADi not found. `pip install casadi`, or set CASADI_LIB / CASADI_INC / CASADI_VER / CASADI_SRC for a CasADi you built yourself — see README)
endif
# Every other input has a usable failure mode; this one does not. With no
# Python to read `casadi.__version__` from, CASADI_VER is empty, the
# version macros are defined as nothing, and the build dies deep in the
# plugin source on `expected primary-expression before ';'` — which names
# neither the option nor the cause. Catch it here instead (gh#634).
ifeq ($(strip $(CASADI_VER)),)
	$(error CASADI_VER is empty. It normally comes from `casadi.__version__`; with no Python available, pass the version of the CasADi you are building against, e.g. CASADI_VER=3.7.2)
endif
	@test -f "$(CASADI_SRC)/casadi/core/nlpsol_impl.hpp" || { \
	  echo "CasADi internal headers not found under CASADI_SRC=$(CASADI_SRC)."; \
	  echo "CASADI_SRC must be the directory *containing* casadi/core/."; \
	  echo "  - pip-installed CasADi: the wheel omits them. Run: make fetch-src"; \
	  echo "  - CasADi built from source: point CASADI_SRC at the source"; \
	  echo "    checkout's root, or rebuild CasADi with"; \
	  echo "    -DINSTALL_INTERNAL_HEADERS=ON (it defaults to OFF) and use"; \
	  echo "    its <prefix>/include for both CASADI_SRC and CASADI_INC."; \
	  exit 1; }
	@test -f "$(POUNCE_LIB)/libpounce_cinterface.$(SO_EXT)" || { \
	  echo "libpounce_cinterface not found in $(POUNCE_LIB)."; \
	  echo "Run: cargo build --release -p pounce-cinterface"; \
	  exit 1; }

## Clone the CasADi source tree at exactly the installed version.
fetch-src:
	@test -n "$(CASADI_VER)" || { echo "CasADi not installed"; exit 1; }
	git clone --depth 1 --branch $(CASADI_VER) \
	  https://github.com/casadi/casadi $(CASADI_SRC)

## Print the flags the installed CasADi was built with, for comparison
## with DEFS above when a plugin fails to load.
abi-flags:
	@echo "installed casadi $(CASADI_VER) at $(CASADI_LIB)"
	@echo "detected _GLIBCXX_USE_CXX11_ABI=$(CXX11_ABI)$(if $(filter Linux,$(UNAME_S)), (from $(CASADI_SO)), (not applicable on $(UNAME_S)))"
	@$(PYTHON) -c "import casadi; print(casadi.CasadiMeta.compiler_flags())"

$(PLUGIN): casadi_nlpsol_pounce.cpp pounce_runtime.hpp convexify_compat.hpp
	$(CXX) -shared $(CXXFLAGS) $(DEFS) $(INCLUDES) $< -o $@ $(LDFLAGS)
	@$(FIXUP)

## Install the plugin next to the CasADi that will load it. CasADi's
## plugin loader searches its own package directory first, so this makes
## `nlpsol(..., 'pounce', ...)` work with no environment variable — and
## it is the user's site-packages, so no sudo.
install: all
	@test -w "$(CASADI_LIB)" || { \
	  echo "$(CASADI_LIB) is not writable — install CasADi in a venv,"; \
	  echo "or skip this and set CASADIPATH=$(CURDIR) instead."; exit 1; }
	cp $(PLUGIN) $(CASADI_LIB)/
	@echo "installed $(PLUGIN) -> $(CASADI_LIB)"

uninstall:
	rm -f $(CASADI_LIB)/$(PLUGIN)

# Worth knowing when an ABI mismatch is what you are chasing: this
# executable, not the plugin, is the one that reports it. A `-shared` link
# leaves undefined symbols to be resolved at load time and succeeds, so the
# plugin builds clean against a CasADi it cannot talk to, and the first
# symptom is CasADi's `Plugin 'pounce' is not found` at dlopen. Linking an
# executable is what turns the same mismatch into a list of names.
$(TEARTEST): $(TEARTEST).cpp $(PLUGIN)
	$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -o $@ $< \
	  -L$(CASADI_LIB) -lcasadi $(RPATH)

test: all $(TEARTEST)
	CASADIPATH=$(CURDIR) $(PYTHON) test_parity.py

examples: all
	@for f in examples/*.py; do \
	  echo "=== $$f"; \
	  CASADIPATH=$(CURDIR) $(PYTHON) $$f || exit 1; \
	done

## Compile the source-compatibility shims against every shape of CasADi
## they have to survive: `convexify_compat.hpp` against both spellings of
## the convexify runtime helper (gh#668), and the plugin's codegen memory
## request against a base class that does and does not declare the virtual
## it binds to (gh#782). Needs no CasADi and no solver library -- mock
## declarations and a compiler -- so both run anywhere, and each covers the
## branch the installed CasADi does not exercise.
compat:
	$(PYTHON) tests/convexify_compat/run.py
	$(PYTHON) tests/codegen_mem_compat/run.py

clean:
	rm -f $(PLUGIN) $(TEARTEST)
