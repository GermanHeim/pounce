// gh#667 regression: POUNCE's log must not tear a line an embedder is
// printing from inside a CasADi callback.
//
// Two writers share file descriptor 1. POUNCE journals from Rust, where
// `io::stdout()` is a `LineWriter` that goes out on every newline; a C++
// embedder writes through `uout()`, which leaves the buffering to
// `std::cout`/stdio — fully buffered behind a pipe. Print a line long
// enough to straddle that buffer from an `iteration_callback` and its tail
// is still pending when the callback returns, so POUNCE's next iteration
// row lands in the middle of it.
//
// This has to be a C++ host: the Python bindings point `Logger::writeFun`
// at `PySys_WriteStdout` but leave `Logger::flush` at `flushDefault`, so
// the plugin's flush cannot reach Python's buffer and the same check run
// from `test_parity.py` would be measuring CasADi, not us. See
// `docs/src/casadi.md`.
//
// Writes the payload in chunks rather than one `<<`: a single write large
// enough to exceed the buffer is pushed straight through and never leaves
// a tail, which is exactly the case that does *not* reproduce.
//
// Prints to stdout; the caller pipes it and counts lines that begin with
// SENTINEL but do not end with TERMINATOR. Pre-fix that count is nonzero.

#include <casadi/casadi.hpp>
#include <string>
#include <vector>

using namespace casadi;

static const char* SENTINEL = "HOST ";
static const char* TERMINATOR = " END";

class IterPrinter : public Callback {
 public:
  casadi_int nx_, ng_, np_;
  IterPrinter(casadi_int nx, casadi_int ng, casadi_int np)
      : nx_(nx), ng_(ng), np_(np) {
    construct("iter_printer");
  }
  casadi_int get_n_in() override { return nlpsol_n_out(); }
  casadi_int get_n_out() override { return 1; }
  std::string get_name_in(casadi_int i) override { return nlpsol_out(i); }
  Sparsity get_sparsity_in(casadi_int i) override {
    std::string n = nlpsol_out(i);
    if (n == "f") return Sparsity::scalar();
    if (n == "x" || n == "lam_x") return Sparsity::dense(nx_);
    if (n == "g" || n == "lam_g") return Sparsity::dense(ng_);
    if (n == "lam_p") return Sparsity::dense(np_);
    return Sparsity(0, 0);
  }
  std::vector<DM> eval(const std::vector<DM>&) const override {
    uout() << SENTINEL;
    for (int k = 0; k < 40; ++k) uout() << std::string(1000, 'A');
    uout() << TERMINATOR << "\n";
    return {0};
  }
};

int main() {
  MX x = MX::sym("x", 2);
  MX f = pow(1 - x(0), 2) + 100 * pow(x(1) - pow(x(0), 2), 2);
  MX g = pow(x(0), 2) + pow(x(1), 2) - 1.5;
  MXDict nlp = {{"x", x}, {"f", f}, {"g", g}};

  IterPrinter cb(2, 1, 0);
  Dict opts;
  opts["iteration_callback"] = cb;
  opts["print_time"] = false;
  // POUNCE's own iteration rows are the competing writer: leave them on.
  opts["pounce"] = Dict{{"print_level", 5}, {"max_iter", 40}};

  Function S = nlpsol("S", "pounce", nlp, opts);
  DMDict arg = {{"x0", std::vector<double>{0.5, 0.5}},
                {"lbg", -inf}, {"ubg", 0}};
  S(arg);
  return 0;
}
