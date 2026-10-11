import math

from arrhenius_fracture.fracture_observation_history import (
    FractureObservation,
    LoadDirection,
    ObservationHistory,
)
from scripts.analyze_current_science_pilots_v2 import event_pairs, kj_record


def observation(sequence, stage, event_id, pre, committed, *, kj=math.nan, ki=math.nan,
                source_row=None):
    return FractureObservation(
        sequence=sequence,
        time_s=float(sequence),
        cycle=0.0,
        segment_id="load-0",
        stage=stage,
        front_id="primary",
        event_id=event_id,
        load_direction=LoadDirection.LOADING,
        pre_length_m=pre,
        committed_length_m=committed,
        K_J_Pa_sqrt_m=kj,
        K_I_Pa_sqrt_m=ki,
        source_row=sequence + 2 if source_row is None else source_row,
    )


def test_kj_is_not_upgraded_to_modal_ki():
    exported = kj_record(observation(0, "pre_event", "e1", 0.0, 0.0, kj=12e6))
    assert exported["K_J_MPa_sqrt_m"] == 12.0
    assert exported["K_I_MPa_sqrt_m"] is None
    assert "J-derived" in exported["K_semantic"]


def test_genuine_modal_ki_remains_a_separate_channel():
    exported = kj_record(
        observation(0, "pre_event", "e1", 0.0, 0.0, kj=12e6, ki=11e6)
    )
    assert exported["K_J_MPa_sqrt_m"] == 12.0
    assert exported["K_I_MPa_sqrt_m"] == 11.0


def test_event_identity_and_stage_come_from_native_history():
    history = ObservationHistory([
        observation(0, "reequilibrated", "", 0.0, 0.0, kj=4e6),
        observation(1, "pre_event", "event-a", 0.0, 0.0, kj=10e6, source_row=3),
        observation(2, "committed", "event-a", 0.0, 5e-6, source_row=3),
        observation(3, "reequilibrated", "", 5e-6, 5e-6, kj=9e6),
    ])
    pairs = event_pairs(history)
    assert len(pairs) == 1
    assert [row.stage.value for row in pairs[0]] == ["pre_event", "committed"]
    assert pairs[0][0].event_id == pairs[0][1].event_id == "event-a"
