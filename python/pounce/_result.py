"""Shared dict/attribute access for SciPy-style result objects.

SciPy's ``solve_ivp`` / ``solve_bvp`` return a ``Bunch`` whose fields are
reachable both as attributes (``res.y``) and as dict items (``res["y"]``).
:class:`ResultMixin` gives pounce's dataclass results the same dual access
so they are drop-in for code written against SciPy.
"""

from __future__ import annotations

import dataclasses


class ResultMixin:
    """Dict-style access for dataclass results, mirroring SciPy's ``Bunch``.

    Adds ``res["field"]`` indexing, ``"field" in res`` membership,
    ``res.keys()`` / ``res.get(...)`` and iteration over field names on top
    of normal attribute access, so a pounce result is interchangeable with a
    SciPy result in downstream code.
    """

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return key in self.keys()

    def keys(self):
        return [f.name for f in dataclasses.fields(self)]

    def __iter__(self):
        return iter(self.keys())

    def get(self, key, default=None):
        return getattr(self, key, default)
