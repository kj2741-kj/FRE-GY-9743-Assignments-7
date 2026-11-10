import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'
    MONOTONE_CONVEX = 'MONOTONE_CONVEX'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        return float(self.values_[idx])

    def integrate(self, start_x: float, end_x: float) -> float:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        overlaps = np.maximum(hi - lo, 0.0)
        return float(np.dot(np.asarray(self.values_, dtype=float), overlaps))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        grad = np.zeros(self.length_)
        grad[idx] = 1.0
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        return np.maximum(hi - lo, 0.0)



class Interpolator1DMonotoneConvex(Interpolator1D):
    """Continuous, node-exact IFR interpolator, monotone and convex on every node interval.

    On each interval [x_i, x_{i+1}] of width h with chord slope s = (v_{i+1} - v_i) / h the
    curve is the quadratic whose derivative is linear, f'(t) = a + (b - a) t / h, so that
        f(x_i + t) = v_i + a t + (b - a) t^2 / (2 h).
    Passing through both endpoints forces the mean slope to equal s, i.e. a + b = 2 s, which
    leaves a single free parameter a (the slope at the left node). Then
        s > 0 : 0 <= a <= s      ->  0 <= a <= b, so f is increasing and convex,
        s < 0 : 2 s <= a <= s    ->  a <= b <= 0, so f is decreasing and convex,
        s = 0 : a = b = 0        ->  f is constant.
    The admissible range is never empty, so every interval is handled, whatever the shape of
    the data. a is picked to match node slope estimates d_i, d_{i+1} in least squares under the
    constraint, then clipped into the range above. The curve is C0; it is C1 at a node whenever
    the clipping is inactive on both sides. FLAT extrapolation holds the end values.
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values, InterpMethod.MONOTONE_CONVEX,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT
        x = np.asarray(self.axis1_, dtype=float)
        v = np.asarray(self.values_, dtype=float)
        n = len(x)
        self.x_, self.v_ = x, v
        self.h_ = np.diff(x)
        self.a_ = np.zeros(max(n - 1, 0))  # slope at the left end of each interval
        self.b_ = np.zeros(max(n - 1, 0))  # slope at the right end of each interval
        self.cum_ = np.zeros(n)            # integral of the curve from x_0 to each node
        if n < 2:
            return

        # Zero-width intervals (repeated nodes) carry no curve; give them zero slope.
        h = self.h_
        s = np.divide(np.diff(v), h, out=np.zeros_like(h), where=h > 0)

        # Node slope estimates: harmonic mean of the adjacent chord slopes when they share a
        # sign, zero when they do not, one-sided at the two ends.
        d = np.empty(n)
        d[0], d[-1] = s[0], s[-1]
        for i in range(1, n - 1):
            if s[i - 1] * s[i] > 0.0:
                d[i] = 2.0 * s[i - 1] * s[i] / (s[i - 1] + s[i])
            else:
                d[i] = 0.0

        a = 0.5 * (d[:-1] + 2.0 * s - d[1:])
        # Admissible range for a: [0, s] if s > 0, [2 s, s] if s < 0, and {0} if s = 0.
        lo = np.where(s > 0, 0.0, 2.0 * s)
        hi = s
        self.a_ = np.clip(a, lo, hi)
        self.b_ = 2.0 * s - self.a_

        # exact integral of each piece: h * (v_i + v_{i+1}) / 2 - h^2 (b - a) / 12
        area = h * (v[:-1] + v[1:]) / 2.0 - h ** 2 * (self.b_ - self.a_) / 12.0
        self.cum_[1:] = np.cumsum(area)

    def _locate(self, x: float) -> int:
        """Index i of the interval [x_i, x_{i+1}] holding x, for x_0 < x < x_{n-1}."""
        i = int(np.searchsorted(self.x_, x, side='right')) - 1
        return min(max(i, 0), self.length_ - 2)

    def interpolate(self, x: float) -> float:
        if self.length_ == 1 or x <= self.x_[0]:
            return float(self.v_[0])
        if x >= self.x_[-1]:
            return float(self.v_[-1])
        i = self._locate(x)
        t = x - self.x_[i]
        return float(self.v_[i] + self.a_[i] * t + (self.b_[i] - self.a_[i]) * t * t / (2.0 * self.h_[i]))

    def _antiderivative(self, x: float) -> float:
        if self.length_ == 1 or x <= self.x_[0]:
            return float(self.v_[0] * (x - self.x_[0]))
        if x >= self.x_[-1]:
            return float(self.cum_[-1] + self.v_[-1] * (x - self.x_[-1]))
        i = self._locate(x)
        t = x - self.x_[i]
        return float(self.cum_[i] + self.v_[i] * t + self.a_[i] * t * t / 2.0
                     + (self.b_[i] - self.a_[i]) * t ** 3 / (6.0 * self.h_[i]))

    def integrate(self, start_x: float, end_x: float) -> float:
        return self._antiderivative(end_x) - self._antiderivative(start_x)

    # Optional risk interface
    # The test cells for risk are not prepared but you can do it by your own.
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        raise NotImplementedError("Optional risk: ordinate gradient is not implemented")

    def gradient_of_integrated_value_wrt_ordinate(
        self, start_x: float, end_x: float,
    ) -> np.ndarray:
        raise NotImplementedError("Optional risk: integrated ordinate gradient is not implemented")


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        elif interpolation_method == InterpMethod.MONOTONE_CONVEX:
            return Interpolator1DMonotoneConvex(axis1_, values_, extrapolation_method)
        else:
            raise ValueError(f'Unsupported interpolation method: {interpolation_method}')
