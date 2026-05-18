<!-- -*- mode: markdown; -*- -->

<div align="center" markdown>
<a name="readme-top"></a>

![peach](https://socialify.git.ci/liblaf/peach/image?description=1&forks=1&issues=1&language=1&logo=https%3A%2F%2Fraw.githubusercontent.com%2Fmicrosoft%2Ffluentui-emoji%2Frefs%2Fheads%2Fmain%2Fassets%2FPeach%2F3D%2Fpeach_3d.png&name=1&owner=1&pattern=Transparent&pulls=1&stargazers=1&theme=Auto)

**[Explore the docs »](https://liblaf.github.io/peach/)**

<!-- tangerine-start: badges/python.md -->

[![codecov](https://codecov.io/gh/liblaf/peach/graph/badge.svg)](https://codecov.io/gh/liblaf/peach)
[![MegaLinter](https://github.com/liblaf/peach/actions/workflows/mega-linter.yaml/badge.svg)](https://github.com/liblaf/peach/actions/workflows/mega-linter.yaml)
[![Test](https://github.com/liblaf/peach/actions/workflows/test.yaml/badge.svg)](https://github.com/liblaf/peach/actions/workflows/test.yaml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/liblaf/peach/main.svg)](https://results.pre-commit.ci/latest/github/liblaf/peach/main)
[![CodSpeed Badge](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/liblaf/peach)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/liblaf-peach?logo=PyPI&label=Downloads)](https://pypi.org/project/liblaf-peach)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/liblaf-peach?logo=Python&label=Python)](https://pypi.org/project/liblaf-peach)
[![PyPI - Version](https://img.shields.io/pypi/v/liblaf-peach?logo=PyPI&label=PyPI)](https://pypi.org/project/liblaf-peach)

<!-- tangerine-end -->

[Documentation](https://liblaf.github.io/peach/) ·
[Changelog](https://github.com/liblaf/peach/blob/main/CHANGELOG.md) ·
[Issues](https://github.com/liblaf/peach/issues)

</div>

## What It Is

Peach is a torch-based toolbox for optimization and linear-solver experiments.
It contains:

- Protocol-based optimizer and linear-system interfaces.
- A preconditioned nonlinear conjugate-gradient optimizer with Armijo
  backtracking and adaptive diagonal Hessian damping.
- CuPy-backed conjugate-gradient and MINRES wrappers for torch tensors, with
  residual diagnostics.
- A SciPy optimizer adapter and a Rosenbrock problem for tests and examples.

## Install

```bash
uv add liblaf-peach
```

## Example

```python
import torch

from liblaf.peach.testing import RosenProblem

problem = RosenProblem()
x = torch.tensor([1.0, 1.0, 1.0])

print(problem.fun(x))
```

## Local Development

```bash
gh repo clone liblaf/peach
cd peach
mise run install
uv run pytest
```

## License

`liblaf-peach` is licensed under the
[MIT License](https://github.com/liblaf/peach/blob/main/LICENSE).
