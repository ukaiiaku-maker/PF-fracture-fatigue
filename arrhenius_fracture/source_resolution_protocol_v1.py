"""Prospective, versioned numerical error budgets; never changes old V3 gates."""
import math

SCHEMA = "v5.source-resolution-final-closure/1"
BASELINE_SHA = "c4fbd2fec207d1a0964db1a02069af62bc0aacde"
MAXIMUM_TOTAL_WAITING_TIME_RELATIVE_ERROR = 0.05
TOTAL_LOG_RATE_BUDGET = math.log1p(MAXIMUM_TOTAL_WAITING_TIME_RELATIVE_ERROR)
ALLOCATED_LOG_RATE_BUDGET = TOTAL_LOG_RATE_BUDGET / 2
ALLOCATED_WAITING_TIME_RELATIVE_ERROR = math.expm1(ALLOCATED_LOG_RATE_BUDGET)
TENSOR_RELATIVE_LIMIT = 0.05
GLOBAL_MINIMUM_QUALITY = 0.05
KIRSCH_RELATIVE_LIMIT = 0.03
TIME_TICK_SECONDS_DECIMAL = "1e-24"
PRACTICAL_LEVELS = ((32, 12), (64, 24), (128, 48))
FINE_REFERENCE = (512, 192)
DEVELOPMENT_SENTINELS = (
    "cavity_only_source", "connected_production_source",
    "poor_shape_negative", "fixed_crack_offset_source",
)
