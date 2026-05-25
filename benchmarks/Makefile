# Benchmark suite Makefile.
#
# This Makefile drives the per-suite benchmarks under benchmarks/.
# The root Makefile delegates to this one for all benchmark targets so there's
# one source of truth. Cargo commands are invoked with --manifest-path pointing
# at the repo root so this Makefile works regardless of the user's CWD.
#
# See benchmarks/README.md for suite descriptions and per-suite READMEs for
# details on each benchmark.

SHELL = /bin/bash
REPO_ROOT := $(shell cd .. && pwd)
CARGO_MANIFEST := $(REPO_ROOT)/Cargo.toml
CARGO := cargo

# ---- libipopt for FFI harnesses (cutest, large_scale) --------------------
# Both crates' build.rs scripts call `pkg-config --libs-only-L ipopt`. The
# default Homebrew install resolves to a MUMPS-linked Ipopt, which mixes
# linear solvers when we benchmark against POUNCE-MA57. Override here to
# pick up the locally-built MA57 install at ref/Ipopt/install-ma57/. The
# CoinHSL pkgconfig dir is also added so `Requires: coinhsl` in ipopt.pc
# resolves. Both build.rs scripts declare `rerun-if-env-changed=PKG_CONFIG_PATH`
# so flipping this variable triggers a rebuild.
IPOPT_MA57_INSTALL ?= $(REPO_ROOT)/ref/Ipopt/install-ma57
COINHSL_DIR        ?= /Users/jkitchin/Dropbox/projects/CoinHSL.v2023.11.17.aarch64-apple-darwin-libgfortran5
IPOPT_MA57_PKGCONFIG := $(IPOPT_MA57_INSTALL)/lib/pkgconfig
COINHSL_PKGCONFIG    := $(COINHSL_DIR)/lib/pkgconfig
# Prepend our two dirs; preserve any caller-supplied PKG_CONFIG_PATH.
export PKG_CONFIG_PATH := $(IPOPT_MA57_PKGCONFIG):$(COINHSL_PKGCONFIG):$(PKG_CONFIG_PATH)
# ipopt-ma57 AMPL executable for .nl-driven suites (electrolyte, grid,
# water, gas, cho, mittelmann). Built by `make -C mittelmann build-ipopt-ma57`.
IPOPT_MA57_BIN     ?= $(IPOPT_MA57_INSTALL)/bin/ipopt

.PHONY: help benchmark benchmark-report \
	check-ipopt-ma57-link build-ipopt-ma57 \
	cutest cutest-ensure-toolchain cutest-prepare cutest-prepare-full cutest-run cutest-full cutest-report cutest-smoke cutest-large cutest-maxiter \
	electrolyte-run grid-run cho-run large-scale mittelmann-run gas-run water-run \
	gams-bench \
	clean-bench clean-bench-cutest clean-bench-large-scale clean-bench-nl clean-bench-report

# CUTEst toolchain auto-install guard. Used as a dependency for every
# cutest-* target so a fresh clone "just works": missing toolchain
# triggers setup_cutest.sh, present toolchain is a no-op.
CUTEST_ROOT_DIR := $(if $(CUTEST_ROOT),$(CUTEST_ROOT),$(HOME)/.local/cutest)
CUTEST_TRAMPOLINE := $(CUTEST_ROOT_DIR)/cutest/src/tools/cutest_trampoline.f90

cutest-ensure-toolchain:
	@if [ ! -f "$(CUTEST_TRAMPOLINE)" ]; then \
		echo "CUTEst toolchain not found at $(CUTEST_ROOT_DIR); running setup_cutest.sh..."; \
		bash $(REPO_ROOT)/benchmarks/cutest/setup_cutest.sh; \
	fi

# Show available targets
help:
	@echo "Benchmark suite targets (run from benchmarks/ or via root Makefile shims)."
	@echo ""
	@echo "  benchmark         Full sweep: CUTEst + electrolyte + grid + large-scale + mittelmann + report + gas + water + gams"
	@echo "  benchmark-report  Regenerate benchmarks/BENCHMARK_REPORT.md"
	@echo ""
	@echo "  cutest            Full CUTEst pipeline: prepare, run, report"
	@echo "  cutest-prepare    Compile SIF problems to .dylib/.so (problem_list.txt, 727)"
	@echo "  cutest-prepare-full  Compile all SIF problems (problem_list_full.txt, 1542)"
	@echo "  cutest-run        Run CUTEst benchmark (writes benchmarks/cutest/results.json)"
	@echo "  cutest-full       Full CUTEst benchmark (all 1542 SIF problems)"
	@echo "  cutest-smoke      Quick smoke test (~20 problems)"
	@echo "  cutest-large      Large problems (n+m >= 100, sparse path)"
	@echo "  cutest-maxiter    Re-run MaxIterations failures"
	@echo "  cutest-report     Generate comparison report from results.json"
	@echo ""
	@echo "  electrolyte-run   Electrolyte thermodynamics benchmark"
	@echo "  grid-run          AC optimal power flow (electrical grid) benchmark"
	@echo "  cho-run           CHO parameter estimation benchmark"
	@echo "  large-scale       Hand-written synthetic large-scale NLPs (pure Rust, no .nl files)"
	@echo "  mittelmann-run    Pounce sweep of the Mittelmann ampl-nlp .nl suite (delegates to benchmarks/mittelmann/Makefile)"
	@echo "  gas-run           Gas pipeline NLPs (AMPL .nl files)"
	@echo "  water-run         Water distribution network NLPs (AMPL .nl files)"
	@echo "  gams-bench        GAMS NLP bench: clean-bench then bench-full (gams/nlpbench)"
	@echo ""
	@echo "  clean-bench       Wipe all stale results/logs across every suite (keeps compiled SIF .dylibs)"
	@echo "  clean-bench-cutest     Wipe cutest results JSON + stderr/log files only"
	@echo "  clean-bench-large-scale Wipe large_scale_results.txt"
	@echo "  clean-bench-nl         Wipe per-problem .sol outputs (gas/water/grid/electrolyte/cho)"
	@echo "  clean-bench-report     Remove benchmarks/BENCHMARK_REPORT.md"
	@echo ""
	@echo "Most suites feed into benchmarks/BENCHMARK_REPORT.md. The gas and water"
	@echo "suites are standalone (see benchmarks/{gas,water}/README.md)."

# ---- ipopt-ma57 plumbing ----
# Verify pkg-config resolves `ipopt` to our MA57 install (not Homebrew).
# Both Rust FFI build.rs scripts use this same pkg-config query.
check-ipopt-ma57-link:
	@echo "PKG_CONFIG_PATH=$$PKG_CONFIG_PATH"
	@libdir=$$(pkg-config --variable libdir ipopt 2>/dev/null); \
	  if [ "$$libdir" != "$(IPOPT_MA57_INSTALL)/lib" ]; then \
	    echo "FAIL: pkg-config resolved ipopt libdir to '$$libdir', expected '$(IPOPT_MA57_INSTALL)/lib'"; \
	    echo "       (did you run 'make build-ipopt-ma57'?)"; exit 1; \
	  fi
	@pkg-config --libs ipopt
	@echo "OK: ipopt FFI route resolves to $(IPOPT_MA57_INSTALL)"

# Convenience shim — the actual recipe lives in benchmarks/mittelmann/Makefile.
build-ipopt-ma57:
	$(MAKE) -C $(REPO_ROOT)/benchmarks/mittelmann build-ipopt-ma57

# ---- CUTEst ----
cutest-prepare: cutest-ensure-toolchain
	bash $(REPO_ROOT)/benchmarks/cutest/prepare.sh

cutest-prepare-full: cutest-ensure-toolchain
	bash $(REPO_ROOT)/benchmarks/cutest/prepare.sh \
		--from-file $(REPO_ROOT)/benchmarks/cutest/problem_list_full.txt

cutest-run: cutest-ensure-toolchain cutest-prepare
	RESULTS_FILE=$(REPO_ROOT)/benchmarks/cutest/results.json \
	$(CARGO) run --manifest-path $(CARGO_MANIFEST) --bin cutest_suite --release \
		2> >(tee $(REPO_ROOT)/benchmarks/cutest/benchmark_stderr.txt >&2)

cutest-full: cutest-ensure-toolchain cutest-prepare-full
	PROBLEM_LIST=$(REPO_ROOT)/benchmarks/cutest/problem_list_full.txt \
	RESULTS_FILE=$(REPO_ROOT)/benchmarks/cutest/full_results.json \
	$(CARGO) run --manifest-path $(CARGO_MANIFEST) --bin cutest_suite --release \
		2> >(tee $(REPO_ROOT)/benchmarks/cutest/full_stderr.txt >&2)

cutest-report:
	python $(REPO_ROOT)/benchmarks/cutest/compare.py $(REPO_ROOT)/benchmarks/cutest/results.json

# Problem lists are defined once and reused by both prepare.sh and the
# cutest_suite binary so they cannot drift.
SMOKE_PROBLEMS := \
	ROSENBR BEALE CUBE DENSCHNB BROWNBS \
	HS6 HS10 HS35 HS71 HS106 \
	MARATOS BT1 HS13 HS57 S316-322 \
	BEALENE BROWNBSNE ENGVAL2NE \
	PALMER1D PALMER5D

LARGE_PROBLEMS := \
	HIMMELBI AIRPORT LAKES SWOPF QPCBLEND QPNBLEND SPANHYD LINSPANH \
	ACOPP14 ACOPR14 ACOPP30 ACOPR30 BATCH CORE1 MSS1 HAIFAM FEEDLOC \
	DISCS DECONVBNE DECONVNE NET1 HYDCAR20 \
	TFI1 TFI2 TFI3 EXPFITB ELATTAR CRESC50 EXPFITC \
	DUALC1 DUALC2 DUALC5 DUALC8 \
	PT HET-Z OET1 OET2 OET3 OET4 KSIP \
	SIPOW1 SIPOW1M SIPOW2 SIPOW2M SIPOW3 SIPOW4 \
	TAXR13322 GOFFIN

MAXITER_PROBLEMS := \
	AIRCRFTB ALLINIT BENNETT5LS BIGGS3 BIGGS5 BOX2 BQPGABIM CHWIRUT2LS \
	DJTL HAHN1LS MEYER3 MGH10LS MINSURF OSBORNEA RAT43LS SIM2BQP THURBERLS \
	AIRCRFTA CONCON DECONVC DECONVNE DECONVU DNIEPER HS109 HS67 HS83 HS84 \
	HS85 HS99EXP MCONCON OPTCNTRL QC QCNEW SSINE TRIGGER

cutest-smoke: cutest-ensure-toolchain
	bash $(REPO_ROOT)/benchmarks/cutest/prepare.sh $(SMOKE_PROBLEMS)
	RESULTS_FILE=$(REPO_ROOT)/benchmarks/cutest/smoke_results.json \
	$(CARGO) run --manifest-path $(CARGO_MANIFEST) --bin cutest_suite --release -- \
		$(SMOKE_PROBLEMS) \
		2> >(tee $(REPO_ROOT)/benchmarks/cutest/smoke_stderr.txt >&2)
	@echo "Smoke test complete."

cutest-large: cutest-ensure-toolchain
	bash $(REPO_ROOT)/benchmarks/cutest/prepare.sh $(LARGE_PROBLEMS)
	RESULTS_FILE=$(REPO_ROOT)/benchmarks/cutest/large_results.json \
	$(CARGO) run --manifest-path $(CARGO_MANIFEST) --bin cutest_suite --release -- \
		$(LARGE_PROBLEMS) \
		2> >(tee $(REPO_ROOT)/benchmarks/cutest/large_stderr.txt >&2)
	@echo "Large problem benchmark complete."

cutest-maxiter: cutest-ensure-toolchain
	bash $(REPO_ROOT)/benchmarks/cutest/prepare.sh $(MAXITER_PROBLEMS)
	RESULTS_FILE=$(REPO_ROOT)/benchmarks/cutest/maxiter_results.json \
	$(CARGO) run --manifest-path $(CARGO_MANIFEST) --bin cutest_suite --release -- \
		$(MAXITER_PROBLEMS) \
		2> >(tee $(REPO_ROOT)/benchmarks/cutest/maxiter_stderr.txt >&2)

cutest: cutest-prepare cutest-run cutest-report
	@echo "CUTEst benchmark complete. Report: benchmarks/cutest/CUTEST_REPORT.md"

# ---- Domain benchmarks ----
# All .nl-file suites use the shared dual-solver driver
# (benchmarks/scripts/run_nl_bench.sh), which runs both pounce and
# ipopt-ma57 on every problem and emits the standard
# {solver,name,n,m,status,objective,iterations,solve_time} JSON shape
# consumed by benchmark_report.py's load_domain_results() loader.
NL_DRIVER       := bash $(REPO_ROOT)/benchmarks/scripts/run_nl_bench.sh
POUNCE_BIN      := $(REPO_ROOT)/target/release/pounce
BENCH_TIMELIMIT ?= 300

# Ensure POUNCE_BIN exists; helps when a target is run before any cargo
# build. Cheap when already built.
$(POUNCE_BIN):
	$(CARGO) build --release --manifest-path $(CARGO_MANIFEST) --bin pounce

electrolyte-run: $(POUNCE_BIN)
	$(NL_DRIVER) electrolyte \
		$(REPO_ROOT)/benchmarks/electrolyte/nl \
		$(REPO_ROOT)/benchmarks/electrolyte/results.json \
		$(POUNCE_BIN) $(IPOPT_MA57_BIN) $(BENCH_TIMELIMIT)

grid-run: $(POUNCE_BIN)
	$(NL_DRIVER) grid \
		$(REPO_ROOT)/benchmarks/grid/nl \
		$(REPO_ROOT)/benchmarks/grid/results.json \
		$(POUNCE_BIN) $(IPOPT_MA57_BIN) $(BENCH_TIMELIMIT)

cho-run: $(POUNCE_BIN)
	$(NL_DRIVER) cho \
		$(REPO_ROOT)/benchmarks/cho/nl_export_results \
		$(REPO_ROOT)/benchmarks/cho/results.json \
		$(POUNCE_BIN) $(IPOPT_MA57_BIN) $(BENCH_TIMELIMIT)

# ---- Large scale ----
# Runs the hand-written synthetic NLP suite (`pounce-large-scale`). Five
# problems — Chained Rosenbrock, Bratu, optimal control, Poisson control,
# sparse QP — exercise the pure-Rust IPM at scales the .nl-file suites
# don't cover. Per-problem sizes can be overridden via env vars; see
# benchmarks/large_scale/src/bin/large_scale_suite.rs.
large-scale:
	$(CARGO) run --manifest-path $(CARGO_MANIFEST) --bin large_scale_suite --release \
		2>&1 | tee $(REPO_ROOT)/benchmarks/large_scale/large_scale_results.txt
	@echo "Large-scale synthetic suite complete. Output: benchmarks/large_scale/large_scale_results.txt"

# ---- Gas pipeline ----
gas-run: $(POUNCE_BIN)
	$(NL_DRIVER) gas \
		$(REPO_ROOT)/benchmarks/gas \
		$(REPO_ROOT)/benchmarks/gas/results.json \
		$(POUNCE_BIN) $(IPOPT_MA57_BIN) $(BENCH_TIMELIMIT)

# ---- Water network ----
water-run: $(POUNCE_BIN)
	$(NL_DRIVER) water \
		$(REPO_ROOT)/benchmarks/water \
		$(REPO_ROOT)/benchmarks/water/results.json \
		$(POUNCE_BIN) $(IPOPT_MA57_BIN) $(BENCH_TIMELIMIT)

# ---- Mittelmann ampl-nlp sweep ----
# Pounce-only sweep delegated to benchmarks/mittelmann/Makefile (the full
# multi-solver comparison harness — fetch / translate / compare across
# pounce, ipopt-mumps, ipopt-ma57, ipopt-feral — lives there; this
# wrapper runs only the pounce variant for inclusion in the composite
# benchmark, same pattern as gams-bench).
mittelmann-run:
	$(MAKE) -C $(REPO_ROOT)/benchmarks/mittelmann run-pounce

# ---- GAMS NLP bench ----
# Wipes cached per-instance trace CSVs/logs and re-runs the full GAMS
# nlpbench sweep (four curated sizes + four model libraries) for every
# solver in REPORT_SOLVERS. Override with `make gams-bench REPORT_SOLVERS="..."`.
gams-bench:
	$(MAKE) -C $(REPO_ROOT)/gams/nlpbench clean-bench bench-full

# ---- Aggregation ----
# Full sweep. Each sub-target runs via `-$(MAKE)` so a single failure
# (missing toolchain, broken instance) doesn't abort the whole run — the
# report at the end always lands. All suites now feed the composite
# BENCHMARK_REPORT.md, so the order is just "everything that produces
# results, then generate the report".
benchmark:
	-$(MAKE) cutest-run
	-$(MAKE) electrolyte-run
	-$(MAKE) grid-run
	-$(MAKE) cho-run
	-$(MAKE) large-scale
	-$(MAKE) mittelmann-run
	-$(MAKE) gas-run
	-$(MAKE) water-run
	-$(MAKE) gams-bench
	-$(MAKE) benchmark-report
	@echo "Full benchmark complete. Report: benchmarks/BENCHMARK_REPORT.md"

# Generate composite report from existing results JSONs.
benchmark-report:
	python $(REPO_ROOT)/benchmarks/benchmark_report.py
	@echo "Report: benchmarks/BENCHMARK_REPORT.md"

# ---- Cleanup ----
# `clean-bench` wipes every suite's outputs and the composite report so the
# next `make benchmark` starts from a clean slate. Compiled SIF .dylibs
# under benchmarks/cutest/problems/ are preserved (expensive to rebuild,
# safe across runs); delete that dir manually for a true cold start.
# Sub-makefiles for mittelmann and gams own their own clean targets and
# are delegated to with `-$(MAKE)` so a missing toolchain there does not
# abort the rest of the cleanup.
clean-bench: clean-bench-cutest clean-bench-large-scale clean-bench-nl clean-bench-report
	-$(MAKE) -C $(REPO_ROOT)/benchmarks/mittelmann clean
	-$(MAKE) -C $(REPO_ROOT)/gams/nlpbench clean-bench
	@echo "clean-bench complete."

clean-bench-cutest:
	rm -f $(REPO_ROOT)/benchmarks/cutest/results.json
	rm -f $(REPO_ROOT)/benchmarks/cutest/results.jsonl
	rm -f $(REPO_ROOT)/benchmarks/cutest/smoke_results.json
	rm -f $(REPO_ROOT)/benchmarks/cutest/large_results.json
	rm -f $(REPO_ROOT)/benchmarks/cutest/maxiter_results.json
	rm -f $(REPO_ROOT)/benchmarks/cutest/full_results.json
	rm -f $(REPO_ROOT)/benchmarks/cutest/benchmark_stderr.txt
	rm -f $(REPO_ROOT)/benchmarks/cutest/smoke_stderr.txt
	rm -f $(REPO_ROOT)/benchmarks/cutest/large_stderr.txt
	rm -f $(REPO_ROOT)/benchmarks/cutest/maxiter_stderr.txt
	rm -f $(REPO_ROOT)/benchmarks/cutest/full_stderr.txt
	rm -f $(REPO_ROOT)/benchmarks/cutest/*.log

clean-bench-large-scale:
	rm -f $(REPO_ROOT)/benchmarks/large_scale/large_scale_results.txt

clean-bench-nl:
	rm -f $(REPO_ROOT)/benchmarks/electrolyte/nl/*.sol \
	      $(REPO_ROOT)/benchmarks/electrolyte/results.json
	rm -f $(REPO_ROOT)/benchmarks/grid/nl/*.sol \
	      $(REPO_ROOT)/benchmarks/grid/results.json
	rm -f $(REPO_ROOT)/benchmarks/gas/*.sol \
	      $(REPO_ROOT)/benchmarks/gas/results.json
	rm -f $(REPO_ROOT)/benchmarks/water/*.sol \
	      $(REPO_ROOT)/benchmarks/water/results.json
	rm -f $(REPO_ROOT)/benchmarks/cho/nl_export_results/*.sol \
	      $(REPO_ROOT)/benchmarks/cho/results.json
	rm -f $(REPO_ROOT)/benchmarks/cho/cho_stderr.txt
	rm -rf $(REPO_ROOT)/benchmarks/logs

clean-bench-report:
	rm -f $(REPO_ROOT)/benchmarks/BENCHMARK_REPORT.md
