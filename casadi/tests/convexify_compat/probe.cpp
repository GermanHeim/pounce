// Compile and run the real `convexify_compat.hpp` shim against mock CasADi
// declarations, in the shape the plugin uses it (gh#668).
//
// Built once per declaration state by `run.py`; see that file and
// `mock_casadi.hpp` for why this is not simply done against a real CasADi.

#include "mock_casadi.hpp"
#include "../../convexify_compat.hpp"

#include <cstdio>

namespace casadi {

  /// Mirrors `PounceInterface::cb_h`'s call: a *non-template* member reached
  /// through a `const` pointer, so `&self->convexify_data_.config` is a
  /// pointer to const, `values` is passed as both `Hin` and `Hout`, and the
  /// work arrays come from the memory object.
  struct FakeInterface {
    ConvexifyData convexify_data_;

    static int cb_h(const FakeInterface* self, double* values, casadi_int* iw,
                    double* w) {
      return convexify_eval_compat(0, &self->convexify_data_.config, values,
                                   values, iw, w);
    }
  };

}  // namespace casadi

int main() {
  casadi::FakeInterface self;
  self.convexify_data_.config.strategy = 2;
  self.convexify_data_.config.margin = 1e-7;

  double values[3] = {1.0, 2.0, 3.0};
  double w[8] = {0.0};
  casadi::casadi_int iw[8] = {0};

  const int rc = casadi::FakeInterface::cb_h(&self, values, iw, w);

  const char* selected = nullptr;
  if (rc == casadi::MOCK_CALLED_NEW) {
    selected = "NEW";
  } else if (rc == casadi::MOCK_CALLED_OLD) {
    selected = "OLD";
  } else {
    std::printf("FAIL: helper returned %d, which is neither sentinel\n", rc);
    return 1;
  }

  // The shim forwards, it does not rearrange. An overload that compiled but
  // swapped `Hin`/`Hout` or dropped a work array would pass a bare
  // "it built" check and then convexify against the wrong buffer.
  const casadi::MockCall& call = casadi::mock_call();
  if (call.c != &self.convexify_data_.config || call.Hin != values ||
      call.Hout != values || call.iw != iw || call.w != w) {
    std::printf("FAIL: arguments did not arrive intact\n");
    return 1;
  }

  std::printf("selected=%s\n", selected);
  return 0;
}
