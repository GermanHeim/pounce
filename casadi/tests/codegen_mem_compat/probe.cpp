// Compile the plugin's `codegen_needs_mem` declaration against mock CasADi
// declarations, in every state a real CasADi can present (gh#782).
//
// The declaration under test is not reproduced here. `run.py` extracts it
// verbatim from `casadi_nlpsol_pounce.cpp` into `plugin_member.inc` and puts
// that file on the include path, so editing the plugin edits the test's
// subject. A copy would drift, and a drifted copy of *this* declaration is
// precisely the defect: the binding it asserts is one the compiler does not
// check.
//
// Built once per declaration state; see `mock_casadi.hpp` for the states and
// `run.py` for what each one must print.

#include "mock_casadi.hpp"

#include <cstdio>

namespace casadi {

  /// Mirrors `PounceInterface`'s shape where the binding is decided: it
  /// derives from `FunctionInternal`, it answers `codegen_mem_type()` with a
  /// non-empty type, and it carries the plugin's own spelling of the
  /// memory request.
  class PounceInterfaceProbe : public FunctionInternal {
  public:
    std::string codegen_mem_type() const override {
      return "struct casadi_pounce_data";
    }

    // Extracted from the plugin by run.py. Deliberately included rather
    // than copied.
    //
    // Both suppressions below are the shadow of the thing under test, not
    // sloppiness, and the plugin carries the first one for the same reason.
    // A member that binds to a base virtual without saying `override` is
    // what `-Winconsistent-missing-override` is for, and a member that
    // fails to bind because the signature moved is what
    // `-Woverloaded-virtual` is for. Each fires in exactly one of the cases
    // below, and this file is compiled `-Werror`, so leaving them on would
    // turn both interesting cases into "did not compile" and lose the
    // distinction the test is here to draw. The verdict comes from
    // `base_call` instead, which separates them by what the program does.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winconsistent-missing-override"
#pragma clang diagnostic ignored "-Woverloaded-virtual"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#endif
#include "plugin_member.inc"
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
  };

}  // namespace casadi

int main() {
  casadi::PounceInterfaceProbe probe;
  const casadi::FunctionInternal* base = &probe;

  // What the plugin's own code sees. True in every state -- this is the
  // uninteresting half, and it is here so that a member that failed to
  // compile is distinguishable from one that answered wrongly.
  std::printf("direct_call=%s\n", probe.codegen_needs_mem() ? "true" : "false");

  // What CasADi sees. This is the whole test: `CodeGenerator` asks through
  // a base-class pointer, so a member that does not override answers
  // `false` here however emphatically it returns `true` above -- and
  // `false` is what stops the `<name>_mem` array being emitted while
  // `codegen_mem()` keeps referring to it.
#if defined(MOCK_HAS_NEEDS_MEM)
  std::printf("base_call=%s\n", base->codegen_needs_mem() ? "true" : "false");
#elif defined(MOCK_NEEDS_MEM_MISMATCH)
  std::printf("base_call=%s\n", base->codegen_needs_mem(0) ? "true" : "false");
#else
  (void) base;
  std::printf("base_call=absent\n");
#endif

  return 0;
}
