// Call CasADi's convexify runtime helper under whichever name the
// installed CasADi declares (gh#668).
//
// CasADi renamed the C++ helper `convexify_eval` to `casadi_convexify_eval`
// after the 3.7.2 release. The codegen SYMBOL name is unchanged, and so are
// the parameters and their order — the break is source-level only, and the
// generated-code path is not affected at all (this plugin refuses to code
// generate `convexify_strategy` in the first place).
//
// The distinguishing fact is not a version number, it is which of the two
// names the installed CasADi declares. Nothing in the version macros
// separates them: the nightly that first carried the rename reports the same
// `CASADI_MAJOR/MINOR/PATCH` as 3.7.2, and keying on `CASADI_IS_RELEASE`
// would then misfire on the next release, which will carry the new name with
// `CASADI_IS_RELEASE=1`. So detect the spelling directly.
//
// Both calls below are dependent on the template parameters, so each is
// looked up at instantiation; the overload naming a helper this CasADi does
// not declare fails substitution in its trailing return type — the immediate
// context — and is dropped rather than becoming a hard error. The unused
// `int` / `long` first parameter exists only to rank the two, so a CasADi
// declaring both spellings resolves to the current name. Call it as
// `convexify_eval_compat(0, ...)`.
//
// Which overload survives is decided by the CasADi being built against, so
// on any one build only one of them is ever compiled. That is what
// `tests/convexify_compat/` is for: it compiles both against mock
// declarations, so the branch this machine's CasADi does not exercise cannot
// rot unnoticed.
//
// Note for anyone patching this downstream: a `sed` that rewrites
// `convexify_eval(` to `casadi_convexify_eval(` also rewrites the *fallback*
// overload's body, leaving a function whose trailing return type names one
// helper and whose body names the other. That compiles by accident — the
// fallback is not instantiated when the new name exists — but it is silently
// wrong, and it is unnecessary now that this header is here.
//
// Requires CasADi's convexify declarations to be included first; this header
// deliberately includes nothing, so the mock-header test can stand in for
// them.
//
// Reported and diagnosed, with a working patch, by @srikanth-gm in gh#668.

#ifndef POUNCE_CASADI_CONVEXIFY_COMPAT_HPP
#define POUNCE_CASADI_CONVEXIFY_COMPAT_HPP

namespace casadi {

  /// CasADi after 3.7.2: the helper carries the `casadi_` prefix.
  template <typename Config, typename T>
  auto convexify_eval_compat(int, const Config* c, const T* Hin, T* Hout,
                             casadi_int* iw, T* w)
      -> decltype(casadi_convexify_eval(c, Hin, Hout, iw, w)) {
    return casadi_convexify_eval(c, Hin, Hout, iw, w);
  }

  /// CasADi 3.6.x and 3.7.x: the unprefixed name. Lower-ranked, so it is
  /// chosen only when the prefixed one is not declared.
  template <typename Config, typename T>
  auto convexify_eval_compat(long, const Config* c, const T* Hin, T* Hout,
                             casadi_int* iw, T* w)
      -> decltype(convexify_eval(c, Hin, Hout, iw, w)) {
    return convexify_eval(c, Hin, Hout, iw, w);
  }

}  // namespace casadi

#endif  // POUNCE_CASADI_CONVEXIFY_COMPAT_HPP
