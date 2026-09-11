"""Analysis-only monotonic action and transient emission/blunting reduction.

Production surfaces are imported unchanged. F1 uses the current persistent-site
source geometry and backstress law with a source-bin moment and the existing
ideal mode-I analytical drive. It is explicitly not the signed spatial F2 model.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
from scipy.special import gammainc, gammaln
from arrhenius_fracture.material_manifest import KB_EV_PER_K
from arrhenius_fracture.inverse_fatigue_barrier_design_v10230 import ideal_mode_I_drive_factors
from arrhenius_fracture.persistent_site_source_v10221 import effective_front_width_m, persistent_site_multiplicity

@dataclass(frozen=True)
class Controls:
    r0: float = 1e-6
    cap: float = 30e9
    hits: float = 3.0
    tau: float = 1e-6
    Kmax: float = 80.0
    rtol: float = 2e-9
    atol: float = 2e-11
    b: float = 2.74e-10
    G: float = 160e9
    L: float = 50e-6
    bins: int = 80
    Lq: float = .5e-6
    line: float = .9124087591240877


def surface(barrier, sigma, T):
    """Exact piecewise EXP-floor derivatives, away from thermal clamp kinks."""
    sigma=np.maximum(np.asarray(sigma,dtype=float),0.)
    rawg=barrier.G00_eV+barrier.gT_eV_per_K*(T-barrier.Tref_K)
    g=max(rawg,1e-12); gt=barrier.gT_eV_per_K if rawg>1e-12 else 0.
    raws=barrier.sigc0_Pa+barrier.sT_Pa_per_K*(T-barrier.Tref_K)
    sc=max(raws,1.); st=barrier.sT_Pa_per_K if raws>1 else 0.
    inner=max(barrier.floor_min_eV,barrier.floor_fraction*g)
    innert=barrier.floor_fraction*gt if barrier.floor_fraction*g>barrier.floor_min_eV else 0.
    f=min(barrier.floor_max_fraction*g,inner)
    ft=barrier.floor_max_fraction*gt if barrier.floor_max_fraction*g<inner else innert
    n=max(barrier.exponent,1e-9); u=max(barrier.alpha,0.)*(sigma/sc)**n
    e=np.exp(-u); value=f+(g-f)*e
    d1=(g-f)*n*u*e
    d2=(g-f)*n*n*u*(1-u)*e
    dt=ft+(gt-ft)*e+(g-f)*e*n*u*st/sc
    return dict(G=value,D1=d1,D2=d2,DT=dt,floor=np.full_like(sigma,f),u=u)


def renewal(barrier,sigma,T,c):
    raw=barrier.rate(sigma,T)
    return gammainc(c.hits,np.minimum(raw*c.tau,1e12))/c.tau


def stress(K,r,c):
    return min(max(K,0.)*1e6/math.sqrt(2*math.pi*r),c.cap)


def f0_action(manifest,K,T,rate,c=Controls()):
    if K==0:return 0.
    # Dimensionless quadrature resolves tiny roots without an imposed load floor.
    return K/rate*quad(lambda x:float(renewal(manifest.cleavage,stress(K*x,c.r0,c),T,c)),0,1,epsabs=1e-13,epsrel=2e-10,limit=250)[0]


def f0_root(manifest,T,rate,xi,c=Controls()):
    if f0_action(manifest,c.Kmax,T,rate,c)<xi:return None
    # Monotone action, root is positive; a logarithmic bracket avoids absolute
    # root tolerances dominating a saturated K of order 1e-9 or smaller.
    hi=c.Kmax;lo=hi
    while f0_action(manifest,lo,T,rate,c)>xi:lo/=10
    return math.exp(brentq(lambda logk:f0_action(manifest,math.exp(logk),T,rate,c)/xi-1,math.log(lo),math.log(hi),xtol=2e-12,rtol=1e-13))


def f0_derivatives(manifest,K,T,rate,xi,c=Controls()):
    barrier=manifest.cleavage
    AK=K*float(renewal(barrier,stress(K,c.r0,c),T,c))/(rate*xi)
    def integrand(x):
        sig=stress(K*x,c.r0,c); d=surface(barrier,sig,T)
        raw=float(barrier.rate(sig,T));arg=raw*c.tau
        # d Lambda/d ln(raw), avoiding division by a vanishing CDF.
        response=math.exp(c.hits*math.log(arg)-arg-gammaln(c.hits))/c.tau if 0<arg<1e4 else 0.
        return response*(float(d['G'])/(KB_EV_PER_K*T*T)-float(d['DT'])/(KB_EV_PER_K*T))
    AT=K/rate/xi*quad(integrand,0,1,epsabs=1e-13,epsrel=2e-10,limit=250)[0]
    return AK,AT


class TransientBlunting:
    """Current source-bin moment, no depletion, recovery, motion, or PT fit.

    N_s is cumulative persistent emission per system. In F1, all emitted line
    remains in the source bins; F2 transport/shielding is deliberately absent.
    Integrate the continuous-time law underlying the production implicit source
    step. The constitutive moment is unchanged as Kdot varies.
    """
    def __init__(self,manifest,row,T,rate,c=Controls()):
        self.m=manifest;self.row=row;self.T=T;self.rate=rate;self.c=c
        dx=c.L/c.bins; L=max(c.Lq,dx,c.b)
        x=(np.arange(c.bins)+.5)*dx
        w=np.exp(-x/L);nsrc=math.ceil(2e-6/dx)
        source_weight=float(w[:nsrc].sum()/nsrc)
        self.qper=c.line*source_weight
        self.rhoper=c.line*source_weight/(w.sum()*dx*max(c.Lq,dx))
        self.factors=ideal_mode_I_drive_factors(30.,.5)
        self.back=c.G*c.b/(1/math.sqrt(3.))
    def state(self,K,N):
        c=self.c;N=np.maximum(N,0.);rho=self.rhoper*N
        r=c.r0+self.m.c_blunt*c.b*self.qper*float(N.sum())
        sig=stress(K,r,c)
        width=effective_front_width_m(5e12+float(rho.sum()),reference_width_m=1e-5,reference_density_m2=5e12,minimum_width_m=c.b,maximum_width_m=c.L)
        mult=persistent_site_multiplicity(float(self.row['rho_source0_m2']),r,width,2.5)
        drive=np.maximum(self.factors*sig-self.back*np.sqrt(rho),0.)
        emitted=np.where(drive>0,mult*self.m.emission.rate(drive,self.T),0.)
        return r,sig,emitted
    def solve(self,Kend,rtol=None):
        c=self.c
        def fun(x,y):
            r,sig,em=self.state(x*Kend,y[:2])
            return np.r_[em*Kend/self.rate,float(renewal(self.m.cleavage,sig,self.T,c))*Kend/self.rate]
        sol=solve_ivp(fun,(0.,1.),np.zeros(3),method='Radau',rtol=rtol or c.rtol,atol=c.atol,dense_output=True,max_step=.025)
        if not sol.success or not np.all(np.isfinite(sol.y)) or np.min(sol.y)<-1e-8:
            raise ValueError('F1_STATE_CLOSURE_UNAVAILABLE: '+sol.message)
        return lambda K:sol.sol(np.asarray(K)/Kend),sol.nfev
    def first_passage(self,xi,guess):
        end=min(self.c.Kmax,guess*1.05)
        for _ in range(30):
            value,nfev=self.solve(end)
            if value(end)[2]>=xi:
                k=brentq(lambda x:value(x*end)[2]/xi-1,0,1,xtol=2e-12)*end
                return k,value,end,nfev
            if end>=self.c.Kmax:return None,value,end,nfev
            end=min(end*2,self.c.Kmax)
        raise ValueError('F1_STATE_CLOSURE_UNAVAILABLE: bracket budget')


def accessibility(zero_fraction,ceiling_fraction,cap_fraction,K,root_residual):
    flags=[]
    if K is None:return 'RAMP_CENSORED',['RAMP_CENSORED']
    if root_residual>1e-6:flags.append('NUMERICAL_LOWER_BOUND_DOMINATED')
    if ceiling_fraction>=.95:flags.append('RENEWAL_CEILING_DOMINATED')
    if zero_fraction>=.99:flags.append('ZERO_LOAD_FIRST_PASSAGE_DOMINATED')
    if cap_fraction>=.95:flags.append('STRESS_CAP_DOMINATED')
    return (flags[0] if flags else 'FRACTURE_RESPONSE_ACCESSIBLE'),flags


def topology(rows,state_available):
    if any(r['accessibility']!='FRACTURE_RESPONSE_ACCESSIBLE' for r in rows):
        return 'FRACTURE_INADMISSIBLE_DESPITE_FATIGUE_CONTROL'
    # Operational numerical descriptors, not experimental material labels.
    K=np.array([r['K_FP'] for r in rows]);d=np.array([r['thermal_derivative'] for r in rows])
    uncertainty=2e-7
    for i in range(1,len(K)-1):
        if d[i-1]>uncertainty and d[i+1]<-uncertainty and K[i]/max(K[0],K[-1])>1.01:
            return 'PEAK_T_FORWARD_PREDICTION'
    if not state_available:return 'UNCLASSIFIED_FORWARD_PREDICTION'
    if np.ptp(np.log(K))<.01 and np.ptp(K)<.01*float(np.median(K)):
        return 'WEAK_T_FORWARD_PREDICTION'
    return 'UNCLASSIFIED_FORWARD_PREDICTION'
