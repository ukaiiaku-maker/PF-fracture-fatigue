"""Declared Ast geometry, protocol and event conventions for initial fitting."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import sqrt

from .cantilever_loading import BendingOpeningShape
from .mechanics import Specimen


@dataclass(frozen=True)
class AstAssumptions:
    B_m: float = 4e-6
    W_m: float = 6e-6
    L_m: float = 12.5e-6
    a0_m: float = 2.25e-6
    r0_m: float = 0.1e-6  # unreported; profiled as a geometry scenario
    mpz_length_m: float = 2e-6
    source_zone_length_m: float = 0.5e-6
    blunting_length_m: float = 0.2e-6
    event_advance_m: float = 20e-9  # declared physical coarse graining
    event_seed: int = 2018001
    target_extension_m: float = 2.3e-6
    maximum_nominal_K_MPa_sqrt_m: float = 45.0
    maximum_load_steps: int = 20000
    nominal_K_step_MPa_sqrt_m: float = 0.02
    localization_bisections: int = 12
    instrument_compliance_m_per_N: float = 0.0
    A2_reference_speed_m_per_s: float = 150e-9
    A2_reference_Kdot_MPa_sqrt_m_per_s: float = 1.6
    A2_reference_E_Pa: float = 411e9  # Table 1 at 21 C; A2 exact T unreported
    loading_mode: str = "published_mode_i"
    bending_opening_shape_path: str | None = None

    def __post_init__(self):
        if not (self.B_m>0 and self.W_m>self.a0_m>0 and self.L_m>0):
            raise ValueError("invalid Ast geometry")
        if not (0<self.r0_m<self.W_m-self.a0_m and
                0<self.event_advance_m<.2e-6 and
                self.event_advance_m<=.1*.2e-6):
            raise ValueError("event length must resolve the 0.2 um onset")
        if self.nominal_K_step_MPa_sqrt_m<=0 or self.localization_bisections<8:
            raise ValueError("invalid numerical localization settings")
        bending_modes = ("nominal_bending_shape",
                         "nominal_bending_shared_temperature")
        if self.loading_mode not in ("published_mode_i", *bending_modes):
            raise ValueError("unknown Ast elastic loading mode")
        if self.loading_mode in bending_modes and not self.bending_opening_shape_path:
            raise ValueError("nominal bending mode requires an elastic field table")
        if self.loading_mode in bending_modes and self.target_extension_m>1e-6:
            raise ValueError("nominal bending table supports at most 1 um extension")

    def specimen(self, condition: dict[str,str]) -> Specimen:
        if (self.loading_mode=="nominal_bending_shape" and
                condition.get("condition_id")!="Tp21C_rate0p5"):
            raise ValueError("nominal bending option is restricted to Tp21C_rate0p5")
        E = float(condition["young_modulus_GPa"])*1e9
        nu = float(condition["poisson_ratio"])
        provisional = Specimen(self.B_m,self.W_m,self.L_m,self.a0_m,E,nu,1e-4)
        C_ref = (provisional.geometry_factor(self.a0_m)*self.A2_reference_speed_m_per_s /
                 (self.A2_reference_Kdot_MPa_sqrt_m_per_s*1e6))
        C_total = C_ref*self.A2_reference_E_Pa/E
        Kdot=float(condition["initial_elastic_Kdot_MPa_sqrt_m_per_s"])*1e6
        C_total=max(C_total,5e-9*provisional.geometry_factor(self.a0_m)/Kdot)
        C_total=min(C_total,150e-9*provisional.geometry_factor(self.a0_m)/Kdot)
        C_specimen = C_total-self.instrument_compliance_m_per_N
        field=(BendingOpeningShape.from_file(self.bending_opening_shape_path)
               if self.loading_mode in ("nominal_bending_shape",
                                        "nominal_bending_shared_temperature")
               else None)
        return replace(provisional,initial_compliance_m_per_N=C_specimen,
                       instrument_compliance_m_per_N=self.instrument_compliance_m_per_N,
                       bending_opening_shape=field)

    def physics(self, base, specimen: Specimen):
        """Set declared Ast geometry; no inherited archived projection table."""
        return replace(base,
            G_Pa=specimen.shear_modulus_Pa,nu=specimen.nu,
            r0_m=self.r0_m,mpz_length_m=self.mpz_length_m,
            source_zone_length_m=self.source_zone_length_m,
            active_strip_width_m=self.blunting_length_m,
            blunting_length_m=self.blunting_length_m,
            reference_source_area_m2=self.B_m*self.mpz_length_m,
            reference_front_width_m=self.B_m,
            maximum_front_width_m=self.B_m,
            emission_schmid_factors=(1/sqrt(3),1/sqrt(3)),
            shielding_orientation_factors=(1.0,-1.0),
            mobile_transport_velocity_scale=1.0,
            emission_geometry_extension_m=(),emission_geometry_factors=())

    def as_dict(self):
        return asdict(self)
