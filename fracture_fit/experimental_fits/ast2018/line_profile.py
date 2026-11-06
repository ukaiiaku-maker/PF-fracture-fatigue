"""Conserved finite-volume line content for the Ast analytical reduction."""
from __future__ import annotations

import numpy as np


def source_birth(profile, activations, conversion, signs, source_bins, cell_area):
    """Inject each activation's declared unit-line content once."""
    acts=np.asarray(activations,float)
    amounts=acts*np.asarray(conversion,float)
    if np.any(amounts< -1e-12) or source_bins<1 or cell_area<=0:
        raise ValueError("invalid source birth")
    for system,(amount,sign) in enumerate(zip(amounts,signs)):
        q=1 if sign>0 else 0
        profile[system,q,:source_bins]+=max(amount,0)/(source_bins*cell_area)
    return amounts


def weighted_density(mobile,retained,dx,strip):
    """Reconstruct the native tip-weighted state without changing its kinetics."""
    x=(np.arange(mobile.shape[-1])+.5)*dx
    weights=np.exp(-x/strip)
    norm=np.sum(weights)
    return (np.sum(mobile*weights,axis=(1,2))/norm,
            np.sum(retained*weights,axis=(1,2))/norm)


def line_content(mobile,retained,cell_area):
    return float(np.sum(mobile+retained)*cell_area)


def accumulated_slip_moment(profile,dx,strip):
    """Native unnormalised tip-weighted line-content moment, by system."""
    x=(np.arange(profile.shape[-1])+.5)*dx
    weights=np.exp(-x/strip)
    return dx*strip*np.sum(np.maximum(profile,0.)*weights,axis=(1,2))


def advance(mobile,retained,velocity,encounter,release,dt,dx,cell_area,
            unit_stress,b_m,front_width_m):
    """Block backward-Euler upwind sweep, with occupancy-consistent reward.

    The signed systems move in opposite x directions. Returned path is the
    physical swept line distance summed over mobile content, not a density.
    """
    if dt<=0:
        return dict(escape=0.,capture=0.,release=0.,path=0.,work=0.,residual=0.)
    before=line_content(mobile,retained,cell_area)
    escape=capture=release_total=path=work=0.
    for system in range(mobile.shape[0]):
        speed=abs(float(velocity[system]))
        flow=speed/dx
        cap=float(encounter[system]);rel=float(release[system])
        for q in range(2):
            order=range(mobile.shape[-1]) if q==1 else range(mobile.shape[-1]-1,-1,-1)
            upstream=0.
            for index in order:
                old_m=float(mobile[system,q,index]);old_r=float(retained[system,q,index])
                a=1+dt*(flow+cap);d=1+dt*rel
                determinant=1+dt*(flow+cap+rel)+dt*dt*flow*rel
                rhs_m=old_m+dt*flow*upstream
                new_m=(d*rhs_m+dt*rel*old_r)/determinant
                new_r=(dt*cap*rhs_m+a*old_r)/determinant
                mobile[system,q,index]=new_m;retained[system,q,index]=new_r
                upstream=new_m
                occupied=dt*new_m*cell_area
                travel=speed*occupied
                path+=travel
                work+=float(unit_stress[system,index])*b_m*front_width_m*travel
                capture+=dt*cap*new_m*cell_area
                release_total+=dt*rel*new_r*cell_area
            escape+=dt*speed*upstream*cell_area/dx
    after=line_content(mobile,retained,cell_area)
    residual=before-after-escape
    if abs(residual)>2e-9*max(before,1e-30):
        raise ArithmeticError("line profile lost content")
    return dict(escape=escape,capture=capture,release=release_total,
                path=path,work=work,residual=residual)


def coupled_source_advance(mobile,retained,velocity,encounter,release,dt,dx,
                           cell_area,unit_stress,b_m,front_width_m,conversion,
                           signs,source_bins,multiplicity,drives,kback,rate_functions,
                           source_root):
    """BE resident response and source root under identical frozen coefficients."""
    base_m=mobile.copy();base_r=retained.copy()
    base=advance(base_m,base_r,velocity,encounter,release,dt,dx,cell_area,
                 unit_stress,b_m,front_width_m)
    base_mobile,base_retained=weighted_density(base_m,base_r,dx,cell_area/dx)
    rho_base=base_mobile+base_retained
    activations=np.zeros(mobile.shape[0])
    totals={key:base[key] for key in ("escape","capture","release","path","work")}
    for system in range(mobile.shape[0]):
        packet=np.zeros_like(mobile)
        one=np.zeros(mobile.shape[0]);one[system]=1.
        source_birth(packet,one,conversion,signs,source_bins,cell_area)
        response_m=packet;response_r=np.zeros_like(packet)
        response=advance(response_m,response_r,velocity,encounter,release,dt,dx,
                         cell_area,unit_stress,b_m,front_width_m)
        response_mobile,response_retained=weighted_density(
            response_m,response_r,dx,cell_area/dx)
        rho_response=(response_mobile+response_retained)[system]
        M=float(np.asarray(multiplicity)[system] if np.ndim(multiplicity) else multiplicity)
        A=(source_root(multiplicity=M,dt_s=dt,
                      drive_stress_Pa=float(drives[system]),
                      rho_initial_m2=float(rho_base[system]),
                      rho_increment_per_activation_m2=float(rho_response),
                      backstress_prefactor_Pa_sqrt_m2=kback,
                      rate_function=rate_functions[system]) if M>0 else 0.)
        activations[system]=A
        base_m+=A*response_m;base_r+=A*response_r
        for key in totals:totals[key]+=A*response[key]
    born=float(np.dot(activations,conversion))
    before=line_content(mobile,retained,cell_area)
    after=line_content(base_m,base_r,cell_area)
    residual=before+born-after-totals["escape"]
    if abs(residual)>2e-9*max(before+born,1e-30):
        raise ArithmeticError("coupled source-to-escape line balance failed")
    mobile[:]=base_m;retained[:]=base_r
    return {**totals,"activations":activations,"birth":born,
            "resident":after,"residual":residual}


def translate_to_new_tip(mobile,retained,distance,dx):
    """Conservatively discard the old-tip slice after crack advance."""
    if distance<=0:return
    cells=distance/dx
    whole=int(cells);fraction=cells-whole
    for profile in (mobile,retained):
        old=profile.copy();profile.fill(0.)
        n=old.shape[-1]
        if whole<n:
            profile[...,:n-whole]+=(1-fraction)*old[...,whole:]
        if whole+1<n:
            profile[...,:n-whole-1]+=fraction*old[...,whole+1:]
