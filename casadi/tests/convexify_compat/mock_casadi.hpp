// Mock stand-in for CasADi's convexify runtime declarations, for the
// compile-time test of `convexify_compat.hpp` (gh#668).
//
// The point of the test is to compile *both* overloads of the shim. A real
// CasADi declares exactly one of the two spellings, so a machine with one
// CasADi installed can only ever compile one of them — and the wheel ships
// builds for casadi 3.6 and 3.7, both of which take the fallback, while
// CI builds against whatever release is current. Mocking the declarations
// is what lets one machine compile every combination.
//
// Faithful to the real header in the things the shim depends on: namespace,
// the config template, `casadi_int`, and the parameter list and its order
// (`const casadi_convexify_config<T1>*, const T1*, T1*, casadi_int*, T1*`,
// returning `int`). Everything else is deliberately absent.
//
// Define exactly one of MOCK_OLD_NAME / MOCK_NEW_NAME to model a released
// CasADi (3.6.x, 3.7.x) or CasADi after 3.7.2; define both to model a
// transitional CasADi carrying an alias; define neither to check that the
// shim fails loudly rather than silently picking something.

#ifndef POUNCE_MOCK_CASADI_HPP
#define POUNCE_MOCK_CASADI_HPP

namespace casadi {

  typedef long long int casadi_int;

  template <typename T1>
  struct casadi_convexify_config {
    int strategy;
    T1 margin;
  };

  /// Sentinels the probe reads back, so the test can tell *which* helper ran
  /// rather than only that the call compiled.
  enum { MOCK_CALLED_OLD = 10, MOCK_CALLED_NEW = 20 };

  /// Set by whichever helper is called, to the arguments it received.
  struct MockCall {
    const void* c = nullptr;
    const void* Hin = nullptr;
    void* Hout = nullptr;
    void* iw = nullptr;
    void* w = nullptr;
  };
  inline MockCall& mock_call() {
    static MockCall call;
    return call;
  }

  template <typename T1>
  inline void mock_record(const casadi_convexify_config<T1>* c, const T1* Hin,
                          T1* Hout, casadi_int* iw, T1* w) {
    MockCall& call = mock_call();
    call.c = c;
    call.Hin = Hin;
    call.Hout = Hout;
    call.iw = iw;
    call.w = w;
  }

#ifdef MOCK_OLD_NAME
  // SYMBOL "convexify_eval"   -- casadi <= 3.7.2
  template <typename T1>
  int convexify_eval(const casadi_convexify_config<T1>* c, const T1* Hin,
                     T1* Hout, casadi_int* iw, T1* w) {
    mock_record(c, Hin, Hout, iw, w);
    return MOCK_CALLED_OLD;
  }
#endif

#ifdef MOCK_NEW_NAME
  // SYMBOL "convexify_eval"   -- casadi after 3.7.2; C++ name prefixed
  template <typename T1>
  int casadi_convexify_eval(const casadi_convexify_config<T1>* c, const T1* Hin,
                            T1* Hout, casadi_int* iw, T1* w) {
    mock_record(c, Hin, Hout, iw, w);
    return MOCK_CALLED_NEW;
  }
#endif

  /// Stands in for `ConvexifyData`, whose `config` member is what the plugin
  /// takes the address of at the call site.
  struct ConvexifyData {
    casadi_convexify_config<double> config;
  };

}  // namespace casadi

#endif  // POUNCE_MOCK_CASADI_HPP
