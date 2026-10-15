"""Conservative parameterized transfer between tip line content and bulk density.

The operator supplies the missing numerical map but intentionally owns no
physical transfer coefficient.  Callers must provide a nonnegative requested
line content and explicit support weights; there is no automatic default.
"""
from __future__ import annotations

import numpy as np


MODEL_ID="conservative-tip-bulk-population-transfer/1"


def _weights(values,name):
    result=np.asarray(values,float)
    if result.ndim!=1 or not np.all(np.isfinite(result)) or np.any(result<0.0):
        raise ValueError(f"{name} must be a finite nonnegative vector")
    if float(np.sum(result))<=0.0:
        raise ValueError(f"{name} must have positive support")
    return result


def conservative_tip_bulk_transfer(
    tip_line_content,bulk_density_m2,bulk_area_m2,*,direction,
    requested_line_content,tip_support_weights,bulk_support_weights,
):
    """Apply one signed-species transfer and return updated arrays plus receipt."""
    tip=np.asarray(tip_line_content,float).copy()
    bulk=np.asarray(bulk_density_m2,float).copy()
    area=np.asarray(bulk_area_m2,float)
    if tip.ndim!=1 or bulk.ndim!=1 or area.shape!=bulk.shape:
        raise ValueError("tip, bulk density and bulk area must be compatible vectors")
    if np.any(~np.isfinite(tip)) or np.any(tip<0.0) or np.any(~np.isfinite(bulk)) or np.any(bulk<0.0):
        raise ValueError("population state must be finite and nonnegative")
    if np.any(~np.isfinite(area)) or np.any(area<=0.0):
        raise ValueError("bulk areas must be positive and finite")
    tip_w=_weights(tip_support_weights,"tip_support_weights")
    bulk_w=_weights(bulk_support_weights,"bulk_support_weights")
    if tip_w.shape!=tip.shape or bulk_w.shape!=bulk.shape:
        raise ValueError("support weights must match their population arrays")
    requested=float(requested_line_content)
    if not np.isfinite(requested) or requested<0.0:
        raise ValueError("requested_line_content must be finite and nonnegative")
    tip_before=float(np.sum(tip)); bulk_before=float(np.sum(bulk*area))
    if direction=="tip_to_bulk":
        capacity=tip*tip_w
        amount=min(requested,float(np.sum(capacity)))
        removal=np.zeros_like(tip) if amount==0.0 else amount*capacity/np.sum(capacity)
        addition_weights=bulk_w/np.sum(bulk_w)
        tip-=removal; bulk+=amount*addition_weights/area
    elif direction=="bulk_to_tip":
        capacity=bulk*area*bulk_w
        amount=min(requested,float(np.sum(capacity)))
        removal_count=np.zeros_like(bulk) if amount==0.0 else amount*capacity/np.sum(capacity)
        addition_weights=tip_w/np.sum(tip_w)
        bulk-=removal_count/area; tip+=amount*addition_weights
    else:
        raise ValueError("direction must be tip_to_bulk or bulk_to_tip")
    tip_after=float(np.sum(tip)); bulk_after=float(np.sum(bulk*area))
    residual=(tip_after+bulk_after)-(tip_before+bulk_before)
    tolerance=1e-12*max(tip_before+bulk_before,amount,1.0)
    if np.any(tip<-tolerance) or np.any(bulk<-tolerance) or abs(residual)>tolerance:
        raise RuntimeError("tip/bulk transfer violated positivity or conservation")
    tip=np.maximum(tip,0.0); bulk=np.maximum(bulk,0.0)
    return tip,bulk,{
        "schema":MODEL_ID,"direction":direction,
        "requested_line_content":requested,"transferred_line_content":amount,
        "limited_by_available_population":amount<requested,
        "tip_count_before":tip_before,"tip_count_after":tip_after,
        "bulk_count_before":bulk_before,"bulk_count_after":bulk_after,
        "conservation_residual_line_content":residual,
        "physical_transfer_coefficient":None,
        "constructed_qualification_parameter":True,
        "production_default":False,
    }


__all__=["MODEL_ID","conservative_tip_bulk_transfer"]
