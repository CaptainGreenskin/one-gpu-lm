"""Explicit generation variants and fail-closed thought/action separation.

The caller cannot supply arbitrary template options. All variants and marker IDs
are part of the hashed serving profile. Raw generation remains in private receipts;
only the action segment is allowed into the tool parser and subsequent history.
"""
import hashlib
import json


def request_options(protocol, arm):
    comparison = protocol.get('reasoning_comparison')
    return {'generation_mode': comparison['arms'][arm]} if comparison else {}


def settings(profile, request):
    modes = profile.get('generation_modes')
    if modes is None:
        if 'generation_mode' in request:
            raise ValueError('Generation variants not enabled by profile')
        return dict(profile['chat_template_kwargs']), 512
    mode = request.get('generation_mode')
    if mode not in modes:
        raise ValueError('Unregistered generation variant')
    cap = profile['generation']['max_new_tokens']
    if type(cap) is not int or not 1 <= cap <= 8192:
        raise ValueError('Invalid profile generation ceiling')
    return dict(modes[mode]), cap


def separate(tokenizer, ids, profile, request):
    options, _ = settings(profile, request)
    marker = profile['reasoning_markers']
    opening, closing = marker['opening_token_id'], marker['closing_token_id']
    if (tokenizer.decode([opening], skip_special_tokens=True) != '<think>'
            or tokenizer.decode([closing], skip_special_tokens=True) != '</think>'):
        raise ValueError('Reasoning marker identity mismatch')
    thinking = options['enable_thinking']
    valid = (opening not in ids and ids.count(closing) == (1 if thinking else 0))
    boundary = ids.index(closing) + 1 if thinking and valid else 0
    # A truncated thought or ambiguous boundary must never reach parse_action.
    action = tokenizer.decode(ids[boundary:], skip_special_tokens=True) if valid else ''
    if '<think>' in action or '</think>' in action:
        valid = False; action = ''
    reason = None if valid else 'missing_or_ambiguous_thought_boundary'
    return {'generation_mode': request['generation_mode'], 'template_kwargs': options,
            'valid': valid, 'error': reason, 'action_text': action,
            'thought_tokens': boundary if valid else (len(ids) if thinking else 0),
            'action_tokens': len(ids) - boundary if valid else 0,
            'unclassified_tokens': 0 if valid or thinking else len(ids),
            'action_sha256': hashlib.sha256(action.encode()).hexdigest()}


def action_text(generated, protocol):
    return generated['separation']['action_text'] if protocol.get('reasoning_comparison') else generated['text']
