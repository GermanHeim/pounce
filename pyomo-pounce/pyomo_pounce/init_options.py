"""One typed options object for the whole initialization stack (gh #609).

Before this module, ``initialize(..., options=...)`` forwarded its dict to
:func:`~pyomo_pounce.project_to_feasible` and to nothing else: the block
solves that follow ran on solver defaults, so a `tol` or `max_iter` tuned
for the model reached the projection and was silently dropped by every
subsystem solve. There was also nowhere to put a knob that is *not* a
solver option -- how the projection merit is scaled, what counts as a
numerically weak block, what to do when a block fails -- so those knobs
did not exist.

:class:`InitOptions` is that object. It carries the solver options once,
plus the initialization policy, and the same instance is threaded through
every stage: projection, block solve, the regularized-least-squares
fallback, and the coupled fallback.

Back compatibility is the reason :meth:`InitOptions.coerce` exists. A
plain mapping has always meant "solver options" at these entry points, and
it still does::

    initialize(m, options={"tol": 1e-8})            # solver options
    initialize(m, options=InitOptions(              # solver options + policy
        solver_options={"tol": 1e-8}, scaling="none"))

A mapping is therefore *never* read as policy, however much its keys look
like field names -- guessing there would silently reinterpret a solver
option (POUNCE has one called ``scaling``) as an initialization policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional

__all__ = ["InitOptions"]

#: Accepted values for :attr:`InitOptions.scaling`.
SCALING_MODES = ("auto", "user", "none")
#: Accepted values for :attr:`InitOptions.conditioning`.
CONDITIONING_MODES = ("auto", "off")
#: Accepted values for :attr:`InitOptions.fallback`.
FALLBACK_MODES = ("regularized", "coupled", "off")
#: Accepted values for :attr:`InitOptions.on_block_failure`.
FAILURE_MODES = ("skip-dependents", "stop")


@dataclass(frozen=True)
class InitOptions:
    """Options for one initialization call, shared by every stage.

    Attributes:
        solver_options: Options handed to **every** solve the pipeline
            runs -- the projection, each subsystem block, and any
            fallback solve. This is what a bare ``options={...}`` dict
            becomes.
        tee: Echo solver output, for every stage.
        scaling: How the projection merit is scaled.

            * ``"auto"`` (default) -- variables are weighted by their own
              magnitude and constraint rows are normalised two-sided, so
              a row in units of 1e6 and a row in units of 1e-6 are
              enforced to the same *relative* accuracy. Entries in the
              model's own ``scaling_factor`` Suffix always win over the
              automatic value.
            * ``"user"`` -- use the ``scaling_factor`` Suffix and nothing
              else; rows and variables the model does not tag stay at 1.
            * ``"none"`` -- the pre-gh#609 behaviour: an unweighted
              ``sum((v - v0)**2)`` merit and no row scaling.
        conditioning: ``"auto"`` (default) runs a numerical rank check on
            each structurally square block before solving it; ``"off"``
            skips the check, as before gh #609.
        cond_tol: A block whose scaled Jacobian has a reciprocal
            condition number below this is *weak* -- structurally square
            but numerically rank-deficient -- and is routed to
            :attr:`fallback` instead of being solved as a square system.
        fallback: What a weak block is routed to.

            * ``"regularized"`` (default) -- minimise the block's scaled
              squared residuals plus ``regularization`` times the scaled
              squared step from the seed. On a rank-deficient block this
              picks the minimum-norm solution of the consistent part
              instead of an arbitrary point on the near-null direction.
            * ``"coupled"`` -- merge the weak block with the blocks that
              depend directly on it and regularize the union, for a
              deficiency that is only resolved by downstream rows.
            * ``"off"`` -- diagnose the block and solve it squarely
              anyway (the pre-gh#609 behaviour, but no longer silent).
        regularization: The ``lambda`` above. Both terms are scaled, so
            this is dimensionless.
        on_block_failure: ``"skip-dependents"`` (default) marks the
            failed block's DAG descendants skipped and carries on with
            independent branches; ``"stop"`` abandons the rest of the
            traversal, as before gh #609.
        max_list: Cap on the reported name lists.
    """

    solver_options: Mapping[str, Any] = field(default_factory=dict)
    tee: bool = False
    scaling: str = "auto"
    conditioning: str = "auto"
    cond_tol: float = 1e-8
    fallback: str = "regularized"
    regularization: float = 1e-8
    on_block_failure: str = "skip-dependents"
    max_list: int = 10

    def __post_init__(self) -> None:
        for name, allowed in (
            ("scaling", SCALING_MODES),
            ("conditioning", CONDITIONING_MODES),
            ("fallback", FALLBACK_MODES),
            ("on_block_failure", FAILURE_MODES),
        ):
            got = getattr(self, name)
            if got not in allowed:
                raise ValueError(
                    f"InitOptions: {name} must be one of "
                    f"{', '.join(map(repr, allowed))}, got {got!r}"
                )
        if self.cond_tol <= 0.0:
            raise ValueError(
                f"InitOptions: cond_tol must be positive, got {self.cond_tol!r}"
            )
        if self.regularization < 0.0:
            raise ValueError(
                "InitOptions: regularization must be non-negative, got "
                f"{self.regularization!r}"
            )
        # Freeze a copy: the caller's dict must not be able to change what
        # a later stage of the same call is handed.
        object.__setattr__(self, "solver_options", dict(self.solver_options))

    @classmethod
    def coerce(cls, options: Optional[Any]) -> "InitOptions":
        """Normalise whatever an entry point was given into `InitOptions`.

        ``None`` -> defaults. An :class:`InitOptions` -> itself. Any other
        mapping -> ``InitOptions(solver_options=dict(options))``, which is
        exactly what ``options={"tol": 1e-8}`` meant before gh #609.
        """
        if options is None:
            return cls()
        if isinstance(options, cls):
            return options
        if isinstance(options, Mapping):
            return cls(solver_options=dict(options))
        raise TypeError(
            "options must be an InitOptions or a mapping of solver options, "
            f"got {type(options).__name__}"
        )

    def solver_kwargs(self) -> Dict[str, Any]:
        """The ``solve()`` keyword arguments every stage passes on.

        One place builds them, so a stage cannot forget one of the two.
        """
        return {"tee": self.tee, "options": dict(self.solver_options)}

    def with_solver_options(self, **extra: Any) -> "InitOptions":
        """A copy whose solver options are these, updated with `extra`.

        Used by the projection to add ``nlp_scaling_method=user-scaling``
        without touching the caller's object -- and it never overrides a
        key the caller set, so an explicit ``nlp_scaling_method`` wins.
        """
        merged = dict(self.solver_options)
        for k, v in extra.items():
            merged.setdefault(k, v)
        return replace(self, solver_options=merged)
