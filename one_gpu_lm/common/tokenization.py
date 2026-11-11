"""One tool-dialogue prefix encoder shared by training and inference."""
from .agent_policy import BASH_TOOLS
from .tool_history import render_history


def template_options(template_kwargs, dialogue_format):
    if dialogue_format not in {'legacy', 'native_schema', 'native_roundtrip'}:
        raise ValueError('Unknown dialogue format')
    options = dict(template_kwargs)
    if dialogue_format != 'legacy':
        options['tools'] = BASH_TOOLS
    return options


def encode_prefix(tokenizer, messages, template_kwargs, dialogue_format):
    rendered = render_history(messages, dialogue_format == 'native_roundtrip')
    dialogue = {'structured_tool_calls': sum(len(m.get('tool_calls', [])) for m in rendered),
                'tool_results': sum(m['role'] == 'tool' for m in rendered)}
    tokens = tokenizer.apply_chat_template(rendered, tokenize=True, return_dict=False,
        add_generation_prompt=True, **template_options(template_kwargs, dialogue_format))
    return tokens, dialogue
