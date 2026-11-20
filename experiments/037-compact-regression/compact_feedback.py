"""Bounded, extractive pytest and TAP failure feedback from public test output."""
from dataclasses import dataclass, field
import re


PYTEST_FAILURES = re.compile(r'^={3,} (?:FAILURES|ERRORS) ={3,}$')
PYTEST_DIVIDER = re.compile(r'^={3,} .+ ={3,}$')
PYTEST_CASE = re.compile(r'^_{3,} \S.*? _{3,}$')
PYTEST_E = re.compile(r'^E\s{2,}(\S.*)$')
PYTEST_SITE = re.compile(r'^(?:[^\s:]+/)*[^\s:]+\.py:\d+: \w+(?:Error|Exception|Exit)\b')
DIFFERENCE = re.compile(r'^([+-])\s*\S')
TAP_CASE = re.compile(r'^not ok\s+\d+\s+\S')
TAP_DIVIDER = re.compile(r'^(?:ok\s+\d+|1\.\.\d+|# (?:tests|fail|pass)\s+|STDERR:)(?:\s|$)')
TAP_ERROR = re.compile(r'^(?:[\w.]+(?:Error|Exception)(?:\s*\[[\w_]+\])?:|error:\s+\S)')
TAP_SITE = re.compile(r'^at .*?(?:\bfile://|/work/|(?:^|[ (])[\w.-]+/)[^\s()]+\.[cm]?js:\d+:\d+')
FIELD_LIMITS = {'case':160, 'failure':235, 'location':140, 'minus':115, 'plus':115}
MAX_CASES = 4
CAP = 1200


@dataclass
class Failure:
    case: str
    fields: dict[str, str] = field(default_factory=dict)

    def first(self, kind, value):
        if value:
            self.fields.setdefault(kind, value)


def pytest_failures(lines):
    failures = []
    section = False
    current = None
    for line in lines:
        if PYTEST_FAILURES.fullmatch(line):
            section = True
            current = None
        elif PYTEST_DIVIDER.fullmatch(line) or line == 'STDERR:':
            section = False
            current = None
        elif section and PYTEST_CASE.fullmatch(line):
            current = Failure(line.strip('_ ').strip())
            failures.append(current)
        elif current is not None:
            match = PYTEST_E.fullmatch(line)
            if match:
                value = match[1]
                diff = DIFFERENCE.match(value)
                current.first(('minus' if diff[1]=='-' else 'plus') if diff else 'failure', value)
            elif PYTEST_SITE.match(line):
                current.first('location', line)
    return failures


def tap_failures(lines):
    failures = []
    current = None
    for line in lines:
        if TAP_CASE.match(line):
            current = Failure(line)
            failures.append(current)
        elif TAP_DIVIDER.match(line):
            current = None
        elif current is not None:
            text = line.strip()
            diff = DIFFERENCE.match(text)
            if TAP_ERROR.match(text):
                current.first('failure', text)
            elif TAP_SITE.match(text):
                current.first('location', text)
            elif diff and not re.fullmatch(r'[-+]\s*(?:expected|actual)(?:\s+[-+]\s*(?:expected|actual))?', text):
                current.first('minus' if diff[1]=='-' else 'plus', text)
    return failures


def extract(output, limit=CAP):
    """Return excerpts and complete source lines for a coverage audit.

    The decoder supplies only visible test output. Unknown output returns None
    so callers can preserve their existing bounded feedback behavior.
    """
    if not 360 <= limit <= CAP:
        raise ValueError('Compact feedback limit outside the supported range')
    lines = output.splitlines()
    if any(PYTEST_FAILURES.fullmatch(line) for line in lines):
        style, failures = 'pytest', pytest_failures(lines)
    elif any(TAP_CASE.match(line) for line in lines):
        style, failures = 'tap', tap_failures(lines)
    else:
        return None
    failures = [case for case in failures if 'failure' in case.fields]
    if not failures:
        return None
    picked = failures[:MAX_CASES]
    count = len(picked)
    omitted = len(failures) - count
    suffix = f'\n[{omitted} additional failure blocks omitted]' if omitted else ''
    per_case = (limit - len(suffix) - 2*(count-1)) // count
    excerpts = []
    exact = []
    for case in picked:
        fields = {'case':case.case, **case.fields}
        labels = {'case':'Test', 'failure':'Failure', 'location':'Location', 'minus':'Diff', 'plus':'Diff'}
        # Give each primary field space before assigning the remainder to diffs.
        core = [name for name in ('case','failure','location') if name in fields]
        chosen = []
        for name in [*core, *(key for key in ('minus','plus') if key in fields)]:
            prefix = labels[name] + ': '
            remaining = per_case - sum(len(line)+1 for line in chosen) - len(prefix)
            if name in core:
                slots = len(core) - len(chosen)
                remaining = min(remaining//slots, FIELD_LIMITS[name])
            else:
                remaining = min(remaining, FIELD_LIMITS[name])
            if remaining < 8:
                break
            text = fields[name]
            shown = text if len(text) <= remaining else text[:remaining-1]+'…'
            chosen.append(prefix+shown)
        excerpts.append('\n'.join(chosen))
        exact.append(fields)
    text = '\n\n'.join(excerpts) + suffix
    if len(text) > limit:
        raise ValueError('Extractor exceeded its character budget')
    return text, {'format':style,'failure_blocks':len(failures),'selected':exact}
