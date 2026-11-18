"""Explicit snapshot identities for historical fixes from the same repository.

Reuse the audited materialization/grading boundary without pretending that
different commits are independent repositories or repository families.
"""
from . import tasks as legacy


def adapt(source, task):
    if (source['source_id'] != task['source_id']
            or source['repository'] != task['repository']
            or source['revision'] != task['revision']
            or source['upstream_parent'] != task['upstream_parent']
            or source['family'] != task['family']):
        raise ValueError('Historical task/source identity mismatch')
    if source['source_id'] != source['repository'].replace('/', '--')+'--'+source['revision']:
        raise ValueError('Snapshot must identify canonical repository and full commit')
    if source['upstream_parent'] == source['revision']:
        raise ValueError('A historical fix requires a distinct parent')
    for revision in (source['upstream_parent'], source['revision']):
        if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
            raise ValueError('Unpinned historical revision')
    s, t = dict(source), dict(task)
    # The adapter's opaque key is used only by the legacy identity check.
    # All reports and split statistics retain canonical repository/family.
    s['repository'] = t['repository'] = source['source_id']
    return s, t


def validate_split(sources, previous_families=()):
    ids = [s['source_id'] for s in sources]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate historical source snapshot')
    adapted = [dict(s, repository=s['source_id'],
                    related_families=[s['repository'],*s.get('related_families',[])]) for s in sources]
    return legacy.validate_split(adapted, previous_families)


def prepare(snapshot, destination, source, task, **kwargs):
    s, t = adapt(source, task)
    if task.get('origin') != 'upstream_history_parent_plus_test_patch':
        raise ValueError('Unexpected historical task construction')
    return legacy.prepare(snapshot, destination, s, t, **kwargs)


def grade(snapshot, destination, source, task, submitted, before, execute):
    s, t = adapt(source, task)
    return legacy.grade(snapshot, destination, s, t, submitted, before, execute)
