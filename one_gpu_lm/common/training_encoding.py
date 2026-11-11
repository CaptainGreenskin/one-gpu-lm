"""Assistant-only labels with a checked boundary against the serving prefix."""
from .tokenization import encode_prefix, template_options
from .tool_history import native_action_message, render_history


def encode_training_example(row, tokenizer, profile):
    dialogue_format = profile['training'].get('dialogue_format', 'legacy')
    if dialogue_format not in {'legacy', 'native_roundtrip'}:
        raise ValueError('Unsupported training dialogue format')
    prefix, dialogue = encode_prefix(tokenizer, row['messages'],
        profile['chat_template_kwargs'], dialogue_format)
    if dialogue_format == 'legacy':
        # Preserve the recorded 011–014 recipe byte-for-byte unless explicitly opted in.
        suffix = tokenizer.encode(row['target'] + tokenizer.eos_token, add_special_tokens=False)
        ids = prefix + suffix
    else:
        rendered = render_history(row['messages'], True)
        rendered.append(native_action_message(row['target']))
        ids = tokenizer.apply_chat_template(rendered, tokenize=True, return_dict=False,
            add_generation_prompt=False,
            **template_options(profile['chat_template_kwargs'], dialogue_format))
        if ids[:len(prefix)] != prefix:
            raise ValueError('Training template does not extend the actual inference prefix')
        suffix = ids[len(prefix):]
        if not suffix:
            raise ValueError('No assistant tokens to supervise')
    return {'ids': ids, 'labels': [-100] * len(prefix) + suffix,
            'supervised_tokens': len(suffix), 'prefix_tokens': len(prefix),
            'dialogue': dialogue, 'dialogue_format': dialogue_format}
