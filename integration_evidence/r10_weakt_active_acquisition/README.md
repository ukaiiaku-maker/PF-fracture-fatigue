# R10 active Weak-T joint acquisition

Joint scientific decision: `RETAIN_ACTIVE_FATIGUE_FRONTIER; NO_JOINT_PROMOTION_AFTER_REFINED_FRACTURE`.

The existing opening-dominated result is preserved at its stated scope.  This
continuation instead starts from the mechanically active Weak-T row and changes
only permitted EXP_floor material coordinates.  Five P4 cold emission knees
were serialized, four received native 15/22 MPa sqrt(m) trajectories, and the
selected 3.0 GPa row received a full five-amplitude native curve with checkpointed
state evolution.  Its developed fatigue metrics are in `analysis/`.

The 3.0 GPa row is a fatigue-qualified active frontier, not a promoted joint
material.  A dK=0.01 fracture calculation exposed the coarse dK=0.05 thermal
screen as a false positive.  Executed P3 hot-opening brackets did not recover
the complete <=1.25 span, so the same-row 2-D gate was not opened.  The package
retains the complete run results and restart checkpoints supporting both the
positive fatigue conclusion and the negative joint decision.
