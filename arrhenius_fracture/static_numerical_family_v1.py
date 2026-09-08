"""Prospective fine-family qualification; original V3 rows remain unchanged."""
import numpy as np
from .closure_mechanics_evidence import REGISTRY as OLD_REGISTRY, GROUPS as OLD_GROUPS
from .finalization_v3_schema import canonical_hash, SCIENTIFIC_ACCEPTANCE_TOLERANCES as LIMITS

SCHEMA = 'v5.quality-constrained-static-families/1'
LEVELS = ((128,48),(256,96),(512,192))
REGISTRY = {}; GROUPS = {}
for label, key in OLD_GROUPS.items():
    if '128:48' not in label: continue
    family = label.replace('128:48', '{mesh}')
    for n, layers in LEVELS:
        cfg = dict(OLD_REGISTRY[key], boundary_segments=n, radial_layers=layers, quality_strategy='flips_and_free_node_optimization')
        new_key = canonical_hash(cfg); REGISTRY[new_key] = cfg
        GROUPS.setdefault(family, {})[str(n)] = new_key


def relative(a, b):
    return float(np.linalg.norm(np.asarray(a)-b)/max(np.linalg.norm(b),1e-300))


def classify(rows, groups=GROUPS):
    output = []
    for family, keys in groups.items():
        peers = [rows[keys[str(n)]] for n,_ in LEVELS]
        if any('failure' in row for row in peers):
            output.append({'family':family, 'source_ids':list(keys.values()), 'passed':False,
                'failure':'REQUIRED_FINE_FAMILY_SOLVE_UNAVAILABLE'}); continue
        measures = [r['measurements'] for r in peers]; first, middle, fine = measures
        gates = {'two_quality_valid_fine_levels':min(middle['mesh_quality'],fine['mesh_quality']) >= .05}
        convergence = {}
        for field, limit in (('reaction',LIMITS['static_mesh_reaction_relative']),
                             ('compliance',LIMITS['static_mesh_reaction_relative']), ('energy',LIMITS['static_mesh_energy_relative'])):
            errors = [relative(a[field],b[field]) for a,b in zip(measures,measures[1:])]
            convergence[field] = errors; gates[field+'_fine_accuracy'] = errors[-1] <= limit
            gates[field+'_stable_limit'] = errors[-1] <= errors[0]
        for field, limit in (('free_residual_relative',LIMITS['free_residual_relative']),
                             ('reaction_balance',LIMITS['reaction_balance_relative']),
                             ('energy_identity',LIMITS['energy_reaction_identity_relative'])):
            gates[field] = all(m[field] <= limit for m in (middle,fine))
        gates['independent_intact_cut'] = all(not m['independent_intact_path_certificate']['intact_cross_graph_path_exists']
            and not m['independent_intact_path_certificate']['insufficient_seed_segment_ids'] for m in (middle,fine))
        if all('tensor_Pa' in m['fixed_tip_probe'] for m in measures):
            errors = [relative(a['fixed_tip_probe']['tensor_Pa'],b['fixed_tip_probe']['tensor_Pa']) for a,b in zip(measures,measures[1:])]
            convergence['fixed_tip_tensor'] = errors
            gates['fixed_tip_fine_accuracy'] = errors[-1] <= LIMITS['tensor_probe_relative']
            gates['fixed_tip_stable_limit'] = errors[-1] <= errors[0]
        else: gates['fixed_tip_fine_accuracy'] = False
        if fine['cavity_enabled']:
            gates['fine_raw_full_boundary_traction'] = fine['cavity_fields'].get('normalized_traction',float('inf')) <= LIMITS['cavity_traction_normalized']
            gates['unique_live_boundary_edge_owners'] = all(m['cavity_fields'].get('edge_owner_valid',False) for m in (middle,fine))
            gates['closed_cavity_and_no_overlap'] = all(m['closed_cavity_boundary_cycle'] and not m['solid_cavity_polygon_overlap_element_ids']
                and not m['support_cavity_polygon_overlap_element_ids'] for m in (middle,fine))
            recovered = [r['recovery'] for r in peers]
            if all(all('recovery' in arc for arc in row) for row in recovered):
                errors = [max(relative(x['recovery']['tensor_Pa'],y['recovery']['tensor_Pa']) for x,y in zip(a,b))
                          for a,b in zip(recovered,recovered[1:])]
                convergence['recovered_fixed_arc_tensor'] = errors
                gates['recovered_fixed_arc_fine_accuracy'] = errors[-1] <= LIMITS['tensor_probe_relative']
                gates['recovered_fixed_arc_stable_limit'] = errors[-1] <= errors[0]
            else: gates['recovered_fixed_arc_fine_accuracy'] = False
        output.append({'family':family,'source_ids':list(keys.values()), 'gates':gates,
            'convergence':convergence,'passed':all(gates.values()), 'coarse_point_quality_retained': first['mesh_quality']})
    derivatives = []
    for kind in ('crack_perturb','radius_perturb'):
        for n,_ in LEVELS[1:]:
            base_key = groups.get('centered:{mesh}',{}).get(str(n))
            if base_key is None: continue
            values = []
            for eps in (2.5e-6,1.25e-6):
                labels = [f'{kind}:{eps}:{sign}:{{mesh}}' for sign in (-1,1)]
                if any(label not in groups for label in labels): continue
                peers = [rows[groups[label][str(n)]] for label in labels]+[rows[base_key]]
                if any('failure' in r for r in peers):
                    derivatives.append({'kind':kind,'level':n,'epsilon':eps,'passed':False,'failure':'DERIVATIVE_PEER_UNAVAILABLE'}); continue
                low,high,base = [r['measurements'] for r in peers]
                dU = -(high['energy']-low['energy'])/(2*eps)
                dC = .5*(4e-7)**2*(high['compliance']-low['compliance'])/(2*eps)/base['compliance']**2
                error = abs(dU-dC)/max(abs(dU),abs(dC),1e-300)
                values.append(dU)
                derivatives.append({'kind':kind,'level':n,'epsilon':eps,'minus_dU':dU,'compliance_derivative':dC,
                    'relative_error':error,'quality_valid_peers':min(m['mesh_quality'] for m in (low,high,base)) >= .05,
                    'passed':min(m['mesh_quality'] for m in (low,high,base)) >= .05 and error <= LIMITS['derivative_energy_compliance_relative']})
            if len(values) == 2:
                error = relative(values[0],values[1])
                derivatives.append({'kind':kind,'level':n,'gate':'perturbation_convergence','relative_error':error,
                    'passed':error <= LIMITS['derivative_perturbation_relative']})
    return {'families':output,'derivatives':derivatives,'passed':bool(output) and all(r['passed'] for r in output+derivatives)}
