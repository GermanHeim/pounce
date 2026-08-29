// Mock stand-in for the part of CasADi's `FunctionInternal` that decides
// whether generated code gets a per-instance memory array (gh#782).
//
// The plugin answers that question with a member function whose binding to
// the base class is *not* checked by the compiler: it cannot be marked
// `override`, because CasADi 3.7 and earlier declare no such virtual and
// `override` is a hard error there. So the member silently overrides on one
// CasADi and silently does not on another, and only one of those two states
// is compiled on any one machine.
//
// That is the same shape as `convexify_compat.hpp`'s problem and it gets the
// same treatment: mock the declaration, compile every state on one machine.
//
// Faithful to the real header in the only things the binding depends on --
// namespace, the class the plugin derives from, and the exact signature
// (`bool codegen_needs_mem() const`, virtual, const-qualified, no
// parameters). Everything else about `FunctionInternal` is deliberately
// absent; nothing here is called for its behaviour.
//
// Define exactly one of:
//
//   MOCK_HAS_NEEDS_MEM       CasADi >= 3.8: the virtual exists, and the
//                            plugin's member must bind to it.
//   MOCK_NEEDS_MEM_MISMATCH  a hypothetical future CasADi that keeps the
//                            name and changes the signature. The plugin's
//                            member then hides rather than overrides, which
//                            is the failure this test exists to make
//                            visible -- it is what an unchecked binding
//                            costs, and the probe reports it rather than
//                            the generated C reporting it three layers
//                            later as `use of undeclared identifier`.
//   (neither)                CasADi <= 3.7: no such virtual. The member must
//                            still compile, which is what forbids `override`.

#ifndef POUNCE_MOCK_CASADI_CODEGEN_MEM_HPP
#define POUNCE_MOCK_CASADI_CODEGEN_MEM_HPP

#include <string>

namespace casadi {

  /// Opaque: the plugin's member takes no arguments and this test never
  /// generates code, so nothing here needs a definition.
  class CodeGenerator;

  class FunctionInternal {
  public:
    virtual ~FunctionInternal() {}

    /// Through CasADi 3.7 this was the whole request: `CodeGenerator`
    /// treated a non-empty return as "emit a memory array for this
    /// function". 3.8 kept it as the *type* and moved the request to
    /// `codegen_needs_mem()`.
    virtual std::string codegen_mem_type() const { return ""; }

#if defined(MOCK_HAS_NEEDS_MEM)
    virtual bool codegen_needs_mem() const { return false; }
#elif defined(MOCK_NEEDS_MEM_MISMATCH)
    virtual bool codegen_needs_mem(int variant) const {
      (void) variant;
      return false;
    }
#endif
  };

}  // namespace casadi

#endif  // POUNCE_MOCK_CASADI_CODEGEN_MEM_HPP
