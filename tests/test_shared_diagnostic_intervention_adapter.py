from types import SimpleNamespace

import numpy as np

from arrhenius_fracture.shared_diagnostic_intervention_adapter import _engine


class FakeAuditedEngine:
    persistent_runtime_binding_records = []

    @classmethod
    def reset_runtime_binding_records(cls):
        cls.persistent_runtime_binding_records = []

    def __init__(self):
        zero = np.zeros((1, 2))
        self.mpz = SimpleNamespace(
            mobile_positive=zero.copy(), mobile_negative=zero.copy(),
            retained_positive=zero.copy(), retained_negative=zero.copy(),
            accumulated_slip_positive=zero.copy(),
            accumulated_slip_negative=zero.copy(),
            wake_mobile_positive=zero.copy(), wake_mobile_negative=zero.copy(),
            wake_retained_positive=zero.copy(), wake_retained_negative=zero.copy(),
            wake_slip_positive=zero.copy(), wake_slip_negative=zero.copy(),
        )
        self.f = SimpleNamespace(r0=1.0e-9)
        self.b = 2.5e-10

    def step(self, K, T, dt):
        self.mpz.retained_positive[0, 0] = 1.0
        return {"accepted": True}


def test_intervention_installs_only_after_first_populated_native_step():
    engine_type = _engine(FakeAuditedEngine, "retained_shielding_off")
    engine_type.reset_runtime_binding_records()
    engine = engine_type()
    assert not hasattr(engine.mpz, "diagnostic_intervention_receipt")
    result = engine.step(1.0, 300.0, 1.0)
    assert result["accepted"] is True
    receipt = engine.mpz.diagnostic_intervention_receipt
    assert receipt["intervention"] == "retained_shielding_off"
    assert receipt["population_state_unchanged"] is True
    assert receipt["applies_from"] == "next_native_step"
    assert receipt["population_measure_at_installation"] == 1.0
    assert len(engine_type.diagnostic_runtime_records) == 1
    engine.step(1.0, 300.0, 1.0)
    assert len(engine_type.diagnostic_runtime_records) == 1
