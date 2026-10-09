"""Exact accepted-time arithmetic with 1e-24 s ticks and lossless subticks.

Inputs retain their actual binary-float values; no decimal rounding or tolerance
is used to manufacture equality. Powers-of-two subdivisions sum exactly to
their original interval. Arbitrary caller partitions that sum to a different
physical duration remain distinguishable.
"""
from dataclasses import dataclass
from fractions import Fraction
import math

TICKS_PER_SECOND = 10**24


def exact(value):
    if isinstance(value, Fraction): return value
    value = float(value)
    if not math.isfinite(value): raise ValueError('finite kinetic scalar required')
    return Fraction.from_float(value)


def packed(value):
    value = exact(value)
    return (value.numerator, value.denominator)


def unpacked(value):
    return Fraction(*value)


@dataclass(frozen=True)
class AcceptedTime:
    ticks: int = 0
    subtick_numerator: int = 0
    subtick_denominator: int = 1

    @classmethod
    def from_seconds(cls, seconds):
        value = exact(seconds)*TICKS_PER_SECOND
        ticks = value.numerator//value.denominator
        remainder = value-ticks
        return cls(ticks, remainder.numerator, remainder.denominator)

    def seconds_exact(self):
        return (self.ticks+Fraction(self.subtick_numerator, self.subtick_denominator))/TICKS_PER_SECOND

    def advance(self, seconds):
        if seconds < 0: raise ValueError('negative accepted interval')
        return self.from_seconds(self.seconds_exact()+exact(seconds))

    def seconds(self):
        return float(self.seconds_exact())


def integrated_constant_rate(initial, rate, elapsed):
    """One final rounding, independent of subdivisions of the accepted interval."""
    if rate < 0 or elapsed < 0: raise ValueError('nonnegative hazard inputs required')
    return exact(initial)+exact(rate)*exact(elapsed)


def integrate_piecewise_linear(anchors, start, end):
    """Exact trapezoidal integral on prescribed physical-time/rate anchors.

    The caller cannot choose quadrature nodes. This helper does not claim that
    a nonlinear FEM-dependent rate is piecewise linear: its owner must provide
    and qualify the actual load/rate path before using this representation.
    """
    points = tuple((exact(t), exact(r)) for t, r in anchors)
    start, end = exact(start), exact(end)
    if (len(points) < 2 or any(r < 0 for _, r in points)
            or any(b[0] <= a[0] for a, b in zip(points, points[1:]))
            or start < points[0][0] or end > points[-1][0] or end < start):
        raise ValueError('invalid physical rate anchors or interval')
    total = Fraction(0)
    for (a, ra), (b, rb) in zip(points, points[1:]):
        lo, hi = max(start, a), min(end, b)
        if hi <= lo: continue
        slope = (rb-ra)/(b-a)
        total += (hi-lo)*(2*ra+slope*(lo+hi-2*a))/2
    return total
