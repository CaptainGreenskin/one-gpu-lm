"""Check reviewed records for known private identifiers and credential patterns."""

import json
import re


SECRET_PATTERNS = {
    'private_key': re.compile(r'-----BEGIN (?:ENCRYPTED |RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----'),
    'github_token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
    'aws_key_id': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'service_token': re.compile(r'\b(?:sk-proj-[A-Za-z0-9_-]{30,}|sk-ant-[A-Za-z0-9_-]{30,}|hf_[A-Za-z0-9]{30,})\b'),
}


def verify_private_labels(records, private_identifiers):
    """Record every checked pattern after the reviewed records pass the scan."""
    for record in records:
        text = json.dumps(record, ensure_ascii=False)
        if (any(pattern.search(text) for pattern in SECRET_PATTERNS.values())
                or any(value and value in text for value in private_identifiers)):
            raise ValueError('The preference export matches a private identifier or credential pattern')
    return dict.fromkeys(SECRET_PATTERNS, 0)
