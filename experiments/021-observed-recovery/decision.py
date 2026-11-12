"""Preregistered candidate selection and fail-closed experimental release gates."""


def paired(base, candidate, expected_ids):
    expected=set(expected_ids)
    def index(value):
        rows=value['results']
        if not value.get('complete') or len(rows)!=len(expected) or {r['id'] for r in rows}!=expected:
            raise ValueError('Require one complete result for every frozen task')
        return {r['id']:r for r in rows}
    a,b=index(base),index(candidate)
    gained=sorted(k for k in expected if not a[k]['passed'] and b[k]['passed'])
    regressed=sorted(k for k in expected if a[k]['passed'] and not b[k]['passed'])
    def format_counts(values):
        generations=[g for r in values.values() for g in r['generations']]
        failures=sum(g['action_format_valid'] is False for g in generations)
        return {'failures':failures,'calls':len(generations),'rate':failures/len(generations) if generations else None}
    return {'pairs':len(expected),'base_passed':sum(r['passed'] for r in a.values()),
        'candidate_passed':sum(r['passed'] for r in b.values()),'gained':gained,'regressed':regressed,
        'net_gain':len(gained)-len(regressed),'base_format':format_counts(a),'candidate_format':format_counts(b),
        'candidate_generated_tokens':sum(r['generated_tokens'] for r in b.values())}


def select(base, reference, observed, expected_ids, plan):
    expected_ids = list(expected_ids)
    if [c['name'] for c in plan['candidates']] != ['observed']:
        raise ValueError('Exactly one new candidate is allowed')
    comparisons = {'base': paired(base, observed, expected_ids),
                   'reference020': paired(reference, observed, expected_ids)}
    rule = plan['selection']
    a, b = comparisons['base'], comparisons['reference020']
    gates = {'base_gain': a['net_gain'] >= rule['must_improve_development_net_tasks'],
             'base_regressions': len(a['regressed']) <= rule['maximum_development_regressions'],
             'reference_gain': b['net_gain'] >= rule['must_improve_reference_net_tasks'],
             'reference_regressions': len(b['regressed']) <= rule['maximum_reference_regressions']}
    return {'selected': 'observed' if all(gates.values()) else None, 'comparisons': comparisons,
            'gates': gates, 'final_test_opened': False, 'selection_uses_development_only': True,
            'reason': 'development_gate_passed' if all(gates.values()) else 'development_gate_failed'}


def release_gate(fresh, historical, checks, plan):
    rule=plan['release_gate']
    format_ok=all(p['base_format']['rate'] is not None and p['candidate_format']['rate'] is not None
                  and p['candidate_format']['rate']<=p['base_format']['rate'] for p in [fresh,historical])
    gates={'fresh_pairs':fresh['pairs']==rule['fresh_tasks'],'historical_pairs':historical['pairs']==rule['historical_tasks'],
        'fresh_gain':fresh['net_gain']>=rule['minimum_net_fresh_gain'],
        'fresh_regressions':len(fresh['regressed'])<=rule['maximum_regressed_fresh_tasks'],
        'historical_regressions':len(historical['regressed'])<=rule['maximum_regressed_historical_tasks'],
        'format_failure_rate':format_ok,'legacy':checks.get('legacy_passes')==rule['legacy_passes_required']}
    for name in ['save_reload_greedy_match','rights_and_secret_scan_pass','all_execution_audits_pass',
                 'single_training_seed_and_repository_cluster_limits_disclosed']:
        gates[name]=checks.get(name) is True
    return {'accepted':all(gates.values()),'gates':gates,'failed_gates':sorted(k for k,v in gates.items() if not v),
            'release_scope':rule['weights_release_scope'],'advanced_coding_or_daily_use_certified':False}
