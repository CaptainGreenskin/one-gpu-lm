"""Serialize executed actions as native tool turns, preserving the private audit log."""
import re

from .actions import parse_action


ACTION_SPANS = re.compile(
    r"```(?:bash|sh)\s*\n(.*?)```|"
    r"<tool_call>\s*<function=bash>\s*<parameter=command>\s*(.*?)\s*</parameter>\s*</function>\s*</tool_call>",
    re.S,
)


def native_action_message(content):
    """Serialize one accepted action; callers establish execution/provenance separately."""
    action = parse_action(content)
    spans = list(ACTION_SPANS.finditer(content))
    if len(spans) != 1:
        raise ValueError('Cannot identify the single accepted action span')
    span = spans[0]
    prose = (content[:span.start()] + content[span.end():]).strip()
    return {'role': 'assistant', 'content': prose, 'tool_calls': [
        {'type': 'function', 'function': {'name': 'bash', 'arguments': dict(action)}}]}


def render_history(messages, native_roundtrip=False):
    """Only an observation carrying executor metadata can become a tool response.

    Re-validate its preceding action with the unchanged parser and match the
    recorded executed command. Format-error feedback stays a user message.
    This performs no execution or repair and never mutates the original history.
    """
    rendered = []
    for index, message in enumerate(messages):
        role, content = message['role'], message['content']
        if role not in {'system', 'user', 'assistant'} or not isinstance(content, str):
            raise ValueError('Unexpected audit message')
        if native_roundtrip and 'tool' in message.get('extra', {}):
            if role != 'user' or not index or messages[index - 1]['role'] != 'assistant':
                raise ValueError('Tool observation requires its preceding assistant action')
            previous = messages[index - 1]
            action = parse_action(previous['content'])
            if previous.get('extra', {}).get('actions') != [action]:
                raise ValueError('Recorded execution and parsed action disagree')
            rendered[-1] = native_action_message(previous['content'])
            rendered.append({'role': 'tool', 'name': 'bash', 'content': content})
        else:
            rendered.append({'role': role, 'content': content})
    return rendered
