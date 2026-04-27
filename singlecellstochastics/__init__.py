"""LAVOUS: lineage-aware variational OU/BM models for scRNA-seq counts.

The distribution name remains ``SingleCellStochastics`` for compatibility with
existing analyses; the public model implemented by the package is LAVOUS.

Public CLI entry points are registered in ``pyproject.toml``:

* ``lavous-heritability`` (``run-plasticity-test``)
* ``lavous-diff`` (``run-diff-test``)
* ``lavous-calibrate`` (``run-calibrate``)
* ``lavous-reconstruct`` (``run-reconst``)
* ``lavous-simulate`` (``run-stochas-sim``)
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
