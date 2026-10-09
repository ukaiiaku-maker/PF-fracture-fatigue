"""V5-only constant-source interval integration from persisted physical anchors."""
from dataclasses import replace
import math
from .canonical_kinetic_time_v1 import exact, packed, unpacked
from .directional_competition_v11 import preview_directional_interval, commit_directional_interval


def advance(hazards, rates, *, clock, anchors, source_signature, maximum_duration_s=None):
    anchors = dict(anchors); prepared = []; crossings = []
    for hazard, rate in zip(hazards, rates):
        value = float(rate['rate_s']); anchor = anchors.get(hazard.candidate_id)
        if value <= 0:
            prepared.append(None); continue
        if (anchor is None or anchor['source_signature'] != source_signature or anchor['rate_s'] != value
                or anchor['expected_action'] != hazard.action
                or anchor['threshold_action'] != hazard.current_threshold_action):
            anchor = {'source_signature': source_signature, 'rate_s': value,
                'initial_hazard': hazard, 'initial_time': clock, 'elapsed': (0, 1),
                'threshold_action': hazard.current_threshold_action, 'expected_action': hazard.action}
        elapsed = unpacked(anchor['elapsed'])
        crossing = (exact(hazard.current_threshold_action)-exact(anchor['initial_hazard'].action))/exact(value)-elapsed
        crossings.append(max(crossing, 0)); prepared.append(anchor)
    if not crossings: return tuple(hazards), tuple(() for _ in hazards), anchors, exact(0)
    duration = min(crossings)
    if maximum_duration_s is not None: duration = min(duration, exact(maximum_duration_s))
    updated = []; events = []
    for hazard, rate, anchor in zip(hazards, rates, prepared):
        if anchor is None or duration == 0:
            updated.append(hazard); events.append(()); continue
        elapsed = unpacked(anchor['elapsed'])+duration
        preview = preview_directional_interval(anchor['initial_hazard'], lambda_per_s=anchor['rate_s'],
            start_time_s=anchor['initial_time'].seconds(), duration_s=float(elapsed))
        fresh = tuple(event for event in preview.completed_events if event.event_ordinal > hazard.completed_event_count)
        # Physical event metadata derives from the common integration anchor,
        # never from the caller's last numerical partition.
        preview = replace(preview, start_action=hazard.action, completed_events=fresh)
        accepted = commit_directional_interval(hazard, preview)
        updated.append(accepted); events.append(fresh)
        anchors[hazard.candidate_id] = {**anchor, 'elapsed': packed(elapsed), 'expected_action': accepted.action}
    return tuple(updated), tuple(events), anchors, duration
