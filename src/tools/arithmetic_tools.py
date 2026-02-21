#!/usr/bin/env python
# coding=utf-8
"""
Deterministic arithmetic tools (addition, subtraction, multiplication, division, power, factorial).

Design goals:
- Mirror the structural style of `poc_tools.py` (clear Tool subclasses + registry via __all__).
- Provide small, fast, side-effect free numeric operations.
- Uniform interface: each tool exposes two inputs (a, b) except Factorial which only needs n.
- Basic validation (types, domain checks for division by zero, factorial domain etc.).
- Return value as string (consistent with many Tool patterns using textual outputs) + numeric value in memory samples.

If desired later, execution-time simulation knobs (noise sampling) can be added similar to retrieval tools.

Noisy variants (relative error model):
- For each deterministic tool, a corresponding noisy tool applies multiplicative noise: result * (1 + eps)
- eps ~ Uniform(-alpha, +alpha), where alpha = 1 - output_quality (clamped to [0,1]).
- Each noisy tool defines an `output_quality` in [0,1]; higher quality => lower alpha => less deviation.
"""
from __future__ import annotations

import math
import random
from typing import Dict, Callable
from tools.base import Tool, BaseArithmeticTool

# ---------------------------------------------------
# Execution time sampling helper (mirrors poc_tools pattern)
# ---------------------------------------------------

def _make_truncated_normal_sampler(
    mu: float,
    sigma: float,
    *,
    min_value: float = 0.0005,
    max_value: float = 0.0500,
) -> Callable[[], float]:
    """Return a sampler for arithmetic tool execution times (very small magnitudes).

    Arithmetic ops are fast; we simulate relative cost differences:
      - accurate tools: slightly higher mean (e.g., validation / precision path)
      - quick tools: lower mean with tighter variance
    """
    def _sample() -> float:
        val = random.gauss(mu, sigma)
        if val < min_value:
            return min_value
        if val > max_value:
            return max_value
        return val

    return _sample

def _make_truncated_lognormal_sampler(mean: float, sigma: float, *,
                                      min_value: float, max_value: float):
    """Return a truncated lognormal sampler (research-scale cost)."""
    m = math.log(max(mean, 1e-6)) - (sigma ** 2) / 2
    def _sample() -> float:
        x = random.lognormvariate(m, sigma)
        return min(max(x, min_value), max_value)
    return _sample

# Global time scaling for arithmetic tools (e.g., simulate 10x faster execution)
TIME_SCALE = 0.1

# ---------------------------------------------------
# Concrete Binary Operation Tools
# ---------------------------------------------------

class AddAccurateTool(BaseArithmeticTool):
    name = "add_verified"
    description = "Addition (verified): uses a modern calculator and double-checks to deliver a perfect result."
    output_quality = 1.0
    # Simulated timing (slower than lightweight/old)
    default_execution_time = 4 * TIME_SCALE  # seconds
    execution_time_mu = 4 * TIME_SCALE
    execution_time_sigma = 0.800 * TIME_SCALE
    execution_time_min = 1.000 * TIME_SCALE
    execution_time_max = 25.000 * TIME_SCALE
    default_execution_time_sampler = _make_truncated_normal_sampler(
        mu=execution_time_mu, sigma=execution_time_sigma, min_value=execution_time_min, max_value=execution_time_max
    )

    cost_mu = 0.01
    cost_sigma = 0.35
    cost_min = 0.006
    cost_max = 0.060
    default_cost = cost_mu
    default_cost_sampler = _make_truncated_lognormal_sampler(
        mean=cost_mu, sigma=cost_sigma, min_value=cost_min, max_value=cost_max
    )

    def _compute(self, a: float, b: float) -> float:
        return a + b


class SubtractAccurateTool(BaseArithmeticTool):
    name = "subtract_verified"
    description = "Subtraction (verified): uses a modern calculator and double-checks to deliver a perfect result."
    output_quality = 1.0
    default_execution_time = 4.000 * TIME_SCALE
    execution_time_mu = 4.000 * TIME_SCALE
    execution_time_sigma = 0.800 * TIME_SCALE
    execution_time_min = 1.000 * TIME_SCALE
    execution_time_max = 25.000 * TIME_SCALE
    default_execution_time_sampler = _make_truncated_normal_sampler(
        mu=execution_time_mu, sigma=execution_time_sigma, min_value=execution_time_min, max_value=execution_time_max
    )

    cost_mu = 0.01
    cost_sigma = 0.35
    cost_min = 0.006
    cost_max = 0.060
    default_cost = cost_mu
    default_cost_sampler = _make_truncated_lognormal_sampler(
        mean=cost_mu, sigma=cost_sigma, min_value=cost_min, max_value=cost_max
    )

    def _compute(self, a: float, b: float) -> float:
        return a - b


class MultiplyAccurateTool(BaseArithmeticTool):
    name = "multiply_verified"
    description = "Multiplication (verified): uses a modern calculator and double-checks to deliver a perfect result."
    output_quality = 1.0
    default_execution_time = 6 * TIME_SCALE
    execution_time_mu = 6 * TIME_SCALE
    execution_time_sigma = 1.000 * TIME_SCALE
    execution_time_min = 1.000 * TIME_SCALE
    execution_time_max = 26.000 * TIME_SCALE
    default_execution_time_sampler = _make_truncated_normal_sampler(
        mu=execution_time_mu, sigma=execution_time_sigma, min_value=execution_time_min, max_value=execution_time_max
    )

    cost_mu = 0.018
    cost_sigma = 0.35
    cost_min = 0.007
    cost_max = 0.070
    default_cost = cost_mu
    default_cost_sampler = _make_truncated_lognormal_sampler(
        mean=cost_mu, sigma=cost_sigma, min_value=cost_min, max_value=cost_max
    )

    def _compute(self, a: float, b: float) -> float:
        return a * b


class DivideAccurateTool(BaseArithmeticTool):
    name = "divide_verified"
    description = "Division (verified): uses a modern calculator and double-checks to deliver a perfect result; raises ValueError on division by zero."
    output_quality = 1.0
    default_execution_time = 7 * TIME_SCALE
    execution_time_mu = 7 * TIME_SCALE
    execution_time_sigma = 1.000 * TIME_SCALE
    execution_time_min = 1.000 * TIME_SCALE
    execution_time_max = 27.000 * TIME_SCALE
    default_execution_time_sampler = _make_truncated_normal_sampler(
        mu=execution_time_mu, sigma=execution_time_sigma, min_value=execution_time_min, max_value=execution_time_max
    )

    cost_mu = 0.020
    cost_sigma = 0.35
    cost_min = 0.008
    cost_max = 0.080
    default_cost = cost_mu
    default_cost_sampler = _make_truncated_lognormal_sampler(
        mean=cost_mu, sigma=cost_sigma, min_value=cost_min, max_value=cost_max
    )

    def _compute(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Division by zero is not allowed.")
        return a / b


class PowerAccurateTool(BaseArithmeticTool):
    name = "power_verified"
    description = "Exponentiation (verified): uses a modern calculator and double-checks to deliver a perfect result."
    output_quality = 1.0
    default_execution_time = 10 * TIME_SCALE
    execution_time_mu = 10 * TIME_SCALE
    execution_time_sigma = 2.200 * TIME_SCALE
    execution_time_min = 1.200 * TIME_SCALE
    execution_time_max = 28.000 * TIME_SCALE
    default_execution_time_sampler = _make_truncated_normal_sampler(
        mu=execution_time_mu, sigma=execution_time_sigma, min_value=execution_time_min, max_value=execution_time_max
    )

    cost_mu = 0.030
    cost_sigma = 0.40
    cost_min = 0.012
    cost_max = 0.120
    default_cost = cost_mu
    default_cost_sampler = _make_truncated_lognormal_sampler(
        mean=cost_mu, sigma=cost_sigma, min_value=cost_min, max_value=cost_max
    )

    def _compute(self, a: float, b: float) -> float:
        return a ** b


# ---------------------------------------------------
# Unary Operation (Factorial)
# ---------------------------------------------------

class FactorialAccurateTool(Tool):
    name = "factorial_verified"
    description = "Factorial (verified): uses a modern calculator, verifies the computation, and returns a perfect n! (0 <= n <= 100)."
    inputs: Dict[str, Dict[str, str]] = {
        "n": {"type": "integer", "description": "Non-negative integer (0 <= n <= 100)."}
    }
    output_type = "string"
    output_quality = 1.0
    default_execution_time = 15 * TIME_SCALE
    execution_time_mu = 15 * TIME_SCALE
    execution_time_sigma = 1.500 * TIME_SCALE
    execution_time_min = 1.500 * TIME_SCALE
    execution_time_max = 28.000 * TIME_SCALE
    default_execution_time_sampler = _make_truncated_normal_sampler(
        mu=execution_time_mu, sigma=execution_time_sigma, min_value=execution_time_min, max_value=execution_time_max
    )

    cost_mu = 0.040
    cost_sigma = 0.45
    cost_min = 0.016
    cost_max = 0.160
    default_cost = cost_mu
    default_cost_sampler = _make_truncated_lognormal_sampler(
        mean=cost_mu, sigma=cost_sigma, min_value=cost_min, max_value=cost_max
    )

    skip_forward_signature_validation = True

    def forward(self, *args, **kwargs) -> str:
        if args:
            n = args[0]
        else:
            n = kwargs.get("n")
        if not isinstance(n, int):
            raise TypeError("Factorial input must be an integer.")
        if n < 0:
            raise ValueError("Factorial is undefined for negative integers.")
        if n > 100:
            raise ValueError("n too large; please use n <= 100 to prevent excessive size.")
        result = math.factorial(n)
        self.memory.add_sample({"n": n, "result": result, "tool": self.name})
        return str(result)


# -----------------------------
# Noisy Relative-Error Variants
# -----------------------------

class _NoisyMixin:
    """Mixin adding relative multiplicative noise based on output_quality.

    Formula:
        alpha = 1 - output_quality (clamped 0..1)
        noisy = exact * (1 + eps), eps ~ Uniform(-alpha, +alpha)
    """
    output_quality: float = 0.8  # default (override per tool if desired)

    def _apply_noise(self, exact: float) -> float:
        q = float(getattr(self, "output_quality", 0.8))
        if q < 0: q = 0
        if q > 1: q = 1
        alpha = 1.0 - q
        if alpha <= 0:
            return exact
        eps = random.uniform(-alpha, alpha)
        return exact * (1.0 + eps)


class AddQuickEstimateTool(_NoisyMixin, AddAccurateTool):
    name = "add_old"
    description = "Lightweight addition [operation_old]: uses an old calculator; fast and simple, somewhat malfunctioning."
    output_quality = 0.95
    default_execution_time = 2.00 * TIME_SCALE
    execution_time_mu = 2.00 * TIME_SCALE
    execution_time_sigma = 0.300 * TIME_SCALE
    execution_time_min = 0.200 * TIME_SCALE
    execution_time_max = 10.000 * TIME_SCALE
    default_execution_time_sampler = _make_truncated_normal_sampler(
        mu=execution_time_mu, sigma=execution_time_sigma, min_value=execution_time_min, max_value=execution_time_max
    )

    cost_mu = 0.009
    cost_sigma = 0.30
    cost_min = 0.004
    cost_max = 0.045
    default_cost = cost_mu
    default_cost_sampler = _make_truncated_lognormal_sampler(
        mean=cost_mu, sigma=cost_sigma, min_value=cost_min, max_value=cost_max
    )

    def _compute(self, a: float, b: float) -> float:
        exact = super()._compute(a, b)
        noisy = self._apply_noise(exact)
        self.memory.add_sample({"a": a, "b": b, "exact": exact, "noisy": noisy, "tool": self.name})
        return noisy


class SubtractQuickEstimateTool(_NoisyMixin, SubtractAccurateTool):
    name = "subtract_old"
    description = "Lightweight subtraction [operation_old]: uses an old calculator; fast and simple, somewhat malfunctioning."
    output_quality = 0.95
    default_execution_time = 2.00 * TIME_SCALE
    execution_time_mu = 2.00 * TIME_SCALE
    execution_time_sigma = 0.400 * TIME_SCALE
    execution_time_min = 0.200 * TIME_SCALE
    execution_time_max = 10.000 * TIME_SCALE
    default_execution_time_sampler = _make_truncated_normal_sampler(
        mu=execution_time_mu, sigma=execution_time_sigma, min_value=execution_time_min, max_value=execution_time_max
    )
    
    cost_mu = 0.009
    cost_sigma = 0.30
    cost_min = 0.004
    cost_max = 0.045
    default_cost = cost_mu
    default_cost_sampler = _make_truncated_lognormal_sampler(
        mean=cost_mu, sigma=cost_sigma, min_value=cost_min, max_value=cost_max
    )

    def _compute(self, a: float, b: float) -> float:
        exact = super()._compute(a, b)
        noisy = self._apply_noise(exact)
        self.memory.add_sample({"a": a, "b": b, "exact": exact, "noisy": noisy, "tool": self.name})
        return noisy


class MultiplyQuickEstimateTool(_NoisyMixin, MultiplyAccurateTool):
    name = "multiply_old"
    description = "Lightweight multiplication [operation_old]: uses an old calculator; fast and simple, somewhat malfunctioning."
    output_quality = 0.95
    default_execution_time = 2.200 * TIME_SCALE
    execution_time_mu = 2.200 * TIME_SCALE
    execution_time_sigma = 0.350 * TIME_SCALE
    execution_time_min = 0.200 * TIME_SCALE
    execution_time_max = 12.000 * TIME_SCALE
    default_execution_time_sampler = _make_truncated_normal_sampler(
        mu=execution_time_mu, sigma=execution_time_sigma, min_value=execution_time_min, max_value=execution_time_max
    )

    cost_mu = 0.011
    cost_sigma = 0.30
    cost_min = 0.005
    cost_max = 0.052
    default_cost = cost_mu
    default_cost_sampler = _make_truncated_lognormal_sampler(
        mean=cost_mu, sigma=cost_sigma, min_value=cost_min, max_value=cost_max
    )

    def _compute(self, a: float, b: float) -> float:
        exact = super()._compute(a, b)
        noisy = self._apply_noise(exact)
        self.memory.add_sample({"a": a, "b": b, "exact": exact, "noisy": noisy, "tool": self.name})
        return noisy


class DivideQuickEstimateTool(_NoisyMixin, DivideAccurateTool):
    name = "divide_old"
    description = "Lightweight division [operation_old]: uses an old calculator; fast and simple, somewhat malfunctioning."
    output_quality = 0.95
    default_execution_time = 3.00 * TIME_SCALE
    execution_time_mu = 3.00 * TIME_SCALE
    execution_time_sigma = 0.400 * TIME_SCALE
    execution_time_min = 0.200 * TIME_SCALE
    execution_time_max = 14.000 * TIME_SCALE
    default_execution_time_sampler = _make_truncated_normal_sampler(
        mu=execution_time_mu, sigma=execution_time_sigma, min_value=execution_time_min, max_value=execution_time_max
    )

    cost_mu = 0.012
    cost_sigma = 0.30
    cost_min = 0.005
    cost_max = 0.058
    default_cost = cost_mu
    default_cost_sampler = _make_truncated_lognormal_sampler(
        mean=cost_mu, sigma=cost_sigma, min_value=cost_min, max_value=cost_max
    )

    def _compute(self, a: float, b: float) -> float:
        exact = super()._compute(a, b)
        noisy = self._apply_noise(exact)
        self.memory.add_sample({"a": a, "b": b, "exact": exact, "noisy": noisy, "tool": self.name})
        return noisy


class PowerQuickEstimateTool(_NoisyMixin, PowerAccurateTool):
    name = "power_old"
    description = "Lightweight exponentiation [operation_old]: uses an old calculator; fast and simple, somewhat malfunctioning."
    output_quality = 0.95
    default_execution_time = 5.0 * TIME_SCALE
    execution_time_mu = 5.0 * TIME_SCALE
    execution_time_sigma = 0.500 * TIME_SCALE
    execution_time_min = 0.300 * TIME_SCALE
    execution_time_max = 16.000 * TIME_SCALE
    default_execution_time_sampler = _make_truncated_normal_sampler(
        mu=execution_time_mu, sigma=execution_time_sigma, min_value=execution_time_min, max_value=execution_time_max
    )

    cost_mu = 0.018
    cost_sigma = 0.32
    cost_min = 0.008
    cost_max = 0.090
    default_cost = cost_mu
    default_cost_sampler = _make_truncated_lognormal_sampler(
        mean=cost_mu, sigma=cost_sigma, min_value=cost_min, max_value=cost_max
    )

    def _compute(self, a: float, b: float) -> float:
        exact = super()._compute(a, b)
        noisy = self._apply_noise(exact)
        self.memory.add_sample({"a": a, "b": b, "exact": exact, "noisy": noisy, "tool": self.name})
        return noisy


class FactorialQuickEstimateTool(_NoisyMixin, FactorialAccurateTool):
    name = "factorial_old"
    description = "Lightweight factorial [operation_old]: uses an old calculator; fast and simple, somewhat malfunctioning."
    output_quality = 0.95
    default_execution_time = 7.5 * TIME_SCALE
    execution_time_mu = 7.5 * TIME_SCALE
    execution_time_sigma = 0.600 * TIME_SCALE
    execution_time_min = 0.300 * TIME_SCALE
    execution_time_max = 18.000 * TIME_SCALE
    default_execution_time_sampler = _make_truncated_normal_sampler(
        mu=execution_time_mu, sigma=execution_time_sigma, min_value=execution_time_min, max_value=execution_time_max
    )

    cost_mu = 0.024
    cost_sigma = 0.35
    cost_min = 0.010
    cost_max = 0.120
    default_cost = cost_mu
    default_cost_sampler = _make_truncated_lognormal_sampler(
        mean=cost_mu, sigma=cost_sigma, min_value=cost_min, max_value=cost_max
    )

    def forward(self, *args, **kwargs) -> str:  # override to insert noise after exact computation
        if args:
            n = args[0]
        else:
            n = kwargs.get("n")
        if not isinstance(n, int):
            raise TypeError("Factorial input must be an integer.")
        if n < 0:
            raise ValueError("Factorial is undefined for negative integers.")
        if n > 100:
            raise ValueError("n too large; please use n <= 100 to prevent excessive size.")
        exact = math.factorial(n)
        noisy = self._apply_noise(float(exact))
        self.memory.add_sample({"n": n, "exact": exact, "noisy": noisy, "tool": self.name})
        return str(noisy)


# -----------------------------
# Very Fast, Lower-Quality Variants
# -----------------------------

class AddFastEstimateTool(_NoisyMixin, AddAccurateTool):
    name = "add_verylightweight"
    description = "Very lightweight addition [operation_verylightweight]: extremely fast; results may be malfunctioning."
    output_quality = 0.80
    default_execution_time = 1.00 * TIME_SCALE
    execution_time_mu = 1.00 * TIME_SCALE
    execution_time_sigma = 0.250 * TIME_SCALE
    execution_time_min = 0.100 * TIME_SCALE
    execution_time_max = 6.000 * TIME_SCALE
    default_execution_time_sampler = _make_truncated_normal_sampler(
        mu=execution_time_mu, sigma=execution_time_sigma, min_value=execution_time_min, max_value=execution_time_max
    )

    cost_mu = 0.005
    cost_sigma = 0.25
    cost_min = 0.002
    cost_max = 0.036
    default_cost = cost_mu
    default_cost_sampler = _make_truncated_lognormal_sampler(
        mean=cost_mu, sigma=cost_sigma, min_value=cost_min, max_value=cost_max
    )

    def _compute(self, a: float, b: float) -> float:
        exact = super()._compute(a, b)
        noisy = self._apply_noise(exact)
        self.memory.add_sample({"a": a, "b": b, "exact": exact, "noisy": noisy, "tool": self.name})
        return noisy


class SubtractFastEstimateTool(_NoisyMixin, SubtractAccurateTool):
    name = "subtract_verylightweight"
    description = "Very lightweight subtraction [operation_verylightweight]: extremely fast; results may be malfunctioning."
    output_quality = 0.80
    default_execution_time = 1.00 * TIME_SCALE
    execution_time_mu = 1.00 * TIME_SCALE
    execution_time_sigma = 0.300 * TIME_SCALE
    execution_time_min = 0.100 * TIME_SCALE
    execution_time_max = 6.000 * TIME_SCALE
    default_execution_time_sampler = _make_truncated_normal_sampler(
        mu=execution_time_mu, sigma=execution_time_sigma, min_value=execution_time_min, max_value=execution_time_max
    )

    cost_mu = 0.005
    cost_sigma = 0.25
    cost_min = 0.002
    cost_max = 0.036
    default_cost = cost_mu
    default_cost_sampler = _make_truncated_lognormal_sampler(
        mean=cost_mu, sigma=cost_sigma, min_value=cost_min, max_value=cost_max
    )

    def _compute(self, a: float, b: float) -> float:
        exact = super()._compute(a, b)
        noisy = self._apply_noise(exact)
        self.memory.add_sample({"a": a, "b": b, "exact": exact, "noisy": noisy, "tool": self.name})
        return noisy


class MultiplyFastEstimateTool(_NoisyMixin, MultiplyAccurateTool):
    name = "multiply_verylightweight"
    description = "Very lightweight multiplication [operation_verylightweight]: extremely fast; results may be malfunctioning."
    output_quality = 0.80
    default_execution_time = 1.20 * TIME_SCALE
    execution_time_mu = 1.20 * TIME_SCALE
    execution_time_sigma = 0.280 * TIME_SCALE
    execution_time_min = 0.120 * TIME_SCALE
    execution_time_max = 8.000 * TIME_SCALE
    default_execution_time_sampler = _make_truncated_normal_sampler(
        mu=execution_time_mu, sigma=execution_time_sigma, min_value=execution_time_min, max_value=execution_time_max
    )

    cost_mu = 0.006
    cost_sigma = 0.25
    cost_min = 0.0025
    cost_max = 0.042
    default_cost = cost_mu
    default_cost_sampler = _make_truncated_lognormal_sampler(
        mean=cost_mu, sigma=cost_sigma, min_value=cost_min, max_value=cost_max
    )

    def _compute(self, a: float, b: float) -> float:
        exact = super()._compute(a, b)
        noisy = self._apply_noise(exact)
        self.memory.add_sample({"a": a, "b": b, "exact": exact, "noisy": noisy, "tool": self.name})
        return noisy


class DivideFastEstimateTool(_NoisyMixin, DivideAccurateTool):
    name = "divide_verylightweight"
    description = "Very lightweight division [operation_verylightweight]: extremely fast; results may be malfunctioning."
    output_quality = 0.80
    default_execution_time = 1.40 * TIME_SCALE
    execution_time_mu = 1.40 * TIME_SCALE
    execution_time_sigma = 0.300 * TIME_SCALE
    execution_time_min = 0.140 * TIME_SCALE
    execution_time_max = 9.000 * TIME_SCALE
    default_execution_time_sampler = _make_truncated_normal_sampler(
        mu=execution_time_mu, sigma=execution_time_sigma, min_value=execution_time_min, max_value=execution_time_max
    )

    cost_mu = 0.007
    cost_sigma = 0.25
    cost_min = 0.003
    cost_max = 0.048
    default_cost = cost_mu
    default_cost_sampler = _make_truncated_lognormal_sampler(
        mean=cost_mu, sigma=cost_sigma, min_value=cost_min, max_value=cost_max
    )

    def _compute(self, a: float, b: float) -> float:
        exact = super()._compute(a, b)
        noisy = self._apply_noise(exact)
        self.memory.add_sample({"a": a, "b": b, "exact": exact, "noisy": noisy, "tool": self.name})
        return noisy


class PowerFastEstimateTool(_NoisyMixin, PowerAccurateTool):
    name = "power_verylightweight"
    description = "Very lightweight exponentiation [operation_verylightweight]: extremely fast; results may be malfunctioning."
    output_quality = 0.80
    default_execution_time = 2.50 * TIME_SCALE
    execution_time_mu = 2.50 * TIME_SCALE
    execution_time_sigma = 0.350 * TIME_SCALE
    execution_time_min = 0.200 * TIME_SCALE
    execution_time_max = 10.000 * TIME_SCALE
    default_execution_time_sampler = _make_truncated_normal_sampler(
        mu=execution_time_mu, sigma=execution_time_sigma, min_value=execution_time_min, max_value=execution_time_max
    )

    cost_mu = 0.010
    cost_sigma = 0.28
    cost_min = 0.004
    cost_max = 0.072
    default_cost = cost_mu
    default_cost_sampler = _make_truncated_lognormal_sampler(
        mean=cost_mu, sigma=cost_sigma, min_value=cost_min, max_value=cost_max
    )

    def _compute(self, a: float, b: float) -> float:
        exact = super()._compute(a, b)
        noisy = self._apply_noise(exact)
        self.memory.add_sample({"a": a, "b": b, "exact": exact, "noisy": noisy, "tool": self.name})
        return noisy


class FactorialFastEstimateTool(_NoisyMixin, FactorialAccurateTool):
    name = "factorial_verylightweight"
    description = "Very lightweight factorial [operation_verylightweight]: extremely fast; results may be malfunctioning."
    output_quality = 0.80
    default_execution_time = 3.50 * TIME_SCALE
    execution_time_mu = 3.50 * TIME_SCALE
    execution_time_sigma = 0.400 * TIME_SCALE
    execution_time_min = 0.200 * TIME_SCALE
    execution_time_max = 10.000 * TIME_SCALE
    default_execution_time_sampler = _make_truncated_normal_sampler(
        mu=execution_time_mu, sigma=execution_time_sigma, min_value=execution_time_min, max_value=execution_time_max
    )

    cost_mu = 0.014
    cost_sigma = 0.30
    cost_min = 0.006
    cost_max = 0.096
    default_cost = cost_mu
    default_cost_sampler = _make_truncated_lognormal_sampler(
        mean=cost_mu, sigma=cost_sigma, min_value=cost_min, max_value=cost_max
    )

    def forward(self, *args, **kwargs) -> str:
        if args:
            n = args[0]
        else:
            n = kwargs.get("n")
        if not isinstance(n, int):
            raise TypeError("Factorial input must be an integer.")
        if n < 0:
            raise ValueError("Factorial is undefined for negative integers.")
        if n > 100:
            raise ValueError("n too large; please use n <= 100 to prevent excessive size.")
        exact = math.factorial(n)
        noisy = self._apply_noise(float(exact))
        self.memory.add_sample({"n": n, "exact": exact, "noisy": noisy, "tool": self.name})
        return str(noisy)


__all__ = [
    # precise (accurate) tools
    "AddAccurateTool",
    "SubtractAccurateTool",
    "MultiplyAccurateTool",
    "DivideAccurateTool",
    "PowerAccurateTool",
    "FactorialAccurateTool",
    # quick approximate variants
    "AddQuickEstimateTool",
    "SubtractQuickEstimateTool",
    "MultiplyQuickEstimateTool",
    "DivideQuickEstimateTool",
    "PowerQuickEstimateTool",
    "FactorialQuickEstimateTool",
    # very fast, lower-quality variants
    "AddFastEstimateTool",
    "SubtractFastEstimateTool",
    "MultiplyFastEstimateTool",
    "DivideFastEstimateTool",
    "PowerFastEstimateTool",
    "FactorialFastEstimateTool",
]

# Backward compatibility aliases (not exported) for previous names
AdditionTool = AddAccurateTool
SubtractionTool = SubtractAccurateTool
MultiplicationTool = MultiplyAccurateTool
DivisionTool = DivideAccurateTool
PowerTool = PowerAccurateTool
FactorialTool = FactorialAccurateTool

add = AddAccurateTool
subtract = SubtractAccurateTool
multiply = MultiplyAccurateTool
divide = DivideAccurateTool
power = PowerAccurateTool
factorial = FactorialAccurateTool

add_noisy = AddQuickEstimateTool
subtract_noisy = SubtractQuickEstimateTool
multiply_noisy = MultiplyQuickEstimateTool
divide_noisy = DivideQuickEstimateTool
power_noisy = PowerQuickEstimateTool
factorial_noisy = FactorialQuickEstimateTool

# convenience aliases for fast variants
add_fast = AddFastEstimateTool
subtract_fast = SubtractFastEstimateTool
multiply_fast = MultiplyFastEstimateTool
divide_fast = DivideFastEstimateTool
power_fast = PowerFastEstimateTool
factorial_fast = FactorialFastEstimateTool
