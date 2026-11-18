"""Select touched upstream Python tests plus deterministic preservation controls.

Selection uses only the pinned source diff, never candidate model outcomes.
Tests and expected values are left unchanged and remain visible to the actor.
"""
import ast
import hashlib


def python_tests(path, text, changed_lines, controls=3):
    tree=ast.parse(text);items=[]
    def collect(body,parents=()):
        for node in body:
            if isinstance(node,ast.ClassDef):
                collect(node.body,(*parents,node.name))
            elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name.startswith('test'):
                start=min([node.lineno]+[d.lineno for d in node.decorator_list])
                items.append({'selector':'::'.join((path,*parents,node.name)),
                              'start':start,'end':node.end_lineno})
    collect(tree.body)
    touched=[x['selector'] for x in items if any(x['start']<=n<=x['end'] for n in changed_lines)]
    if not touched:
        return {'selectors':[path],'touched':[],'controls':[],
                'mode':'whole_module_fallback_no_changed_collectable_test_function'}
    remaining=[x['selector'] for x in items if x['selector'] not in set(touched)]
    ranked=sorted(remaining,key=lambda s:hashlib.sha256(('032-upstream-controls-v1\0'+s).encode()).hexdigest())
    selected_controls=ranked[:controls]
    return {'selectors':touched+selected_controls,'touched':touched,'controls':selected_controls,
            'mode':'changed_test_functions_and_fixed_hash_controls'}


def regression_wrapper(language, selectors, preload=None):
    if not selectors or language not in {'python','javascript'}:raise ValueError('Invalid regression selection')
    if language=='python':
        args=['-q','-p','no:cacheprovider','-o','addopts=','-o','required_plugins=',
              '--disable-warnings','--tb=short',*selectors]
        return '"""Pinned upstream test selection; expected assertions are unchanged."""\nimport pytest\nraise SystemExit(pytest.main('+repr(args)+'))\n'
    args=['node']
    if preload:args+=['--require',preload]
    args+=['--test','--test-reporter=tap',*selectors]
    return '"""Run public upstream JavaScript checks without a shell."""\nimport subprocess\nraise SystemExit(subprocess.run('+repr(args)+').returncode)\n'
