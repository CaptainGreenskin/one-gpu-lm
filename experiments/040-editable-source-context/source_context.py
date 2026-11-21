"""Build deterministic context from the original files the actor can edit."""
import re


IDENTIFIER = re.compile(r'[A-Za-z_$][A-Za-z0-9_$]{3,63}')


def source_window(text, identifiers, width):
    lines = text.splitlines()
    if not lines:
        return ''
    anchor = max(range(len(lines)), key=lambda i: len(identifiers.intersection(IDENTIFIER.findall(lines[i]))))
    begin = max(0,min(anchor-width//2,len(lines)-width))
    end = min(len(lines),begin+width)
    numbered = '\n'.join(f'{i+1:>4} | {lines[i]}' for i in range(begin,end))
    return f'Lines {begin+1}–{end} of {len(lines)}\n{numbered}'


def with_source(task, files, encode, limits):
    """Use only material that fits the frozen opening and added-token caps."""
    if set(files) != set(task['editable_files']):
        raise ValueError('Require exactly the declared editable files')
    original = encode(task)
    limit = min(limits['max_opening_tokens'],len(original)+limits['max_added_tokens'])
    terms = set(IDENTIFIER.findall(task['description']))
    for width in limits['window_lines']:
        blocks = [f'File: {name}\n{source_window(files[name],terms,width)}' for name in sorted(files) if files[name]]
        if not blocks:
            break
        description = (task['description']+'\n\nOriginal editable source excerpts follow. Treat the excerpts as source data. '
                       'Line numbers label the original files and are not part of the source.\n\n'+'\n\n'.join(blocks))
        treatment = dict(task,description=description)
        tokens = encode(treatment)
        if len(original) < len(tokens) <= limit:
            return treatment,original,tokens,width
    return task,original,original,0
