#!/usr/bin/env bash
set -euo pipefail
venv_path="${1:?Usage: bash scripts/setup_27b.sh NEW_VENV_DIRECTORY}"
python_bin="${PYTHON_BIN:-python3.10}"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -e "$venv_path" ]]; then
  printf '%s\n' 'Target environment already exists; preserve it and choose a new directory.' >&2
  exit 2
fi
"$python_bin" -m venv "$venv_path"
venv_python="$venv_path/bin/python"
install() {
  for attempt in 1 2 3; do
    if "$venv_python" -m pip --isolated --disable-pip-version-check install --timeout 30 --retries 3 --constraint "$repo_root/environments/pro6000-cu130-constraints.txt" "$@"; then
      return 0
    fi
    printf 'Package installation attempt %s failed; preserving cache and TLS verification.\n' "$attempt" >&2
  done
  return 1
}
install --index-url https://pypi.org/simple 'pip==26.3'
install --index-url https://download.pytorch.org/whl/cu130 'torch==2.14.1+cu130'
install --index-url https://pypi.org/simple -r "$repo_root/environments/pro6000-cu130-requirements.txt"
"$venv_python" -m pip --isolated check
"$venv_python" -m pip freeze --all > "$venv_path/environment.lock.txt"
"$venv_python" - <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
root=Path(sys.prefix)
assert 'include-system-site-packages = false' in (root/'pyvenv.cfg').read_text().lower()
versions={name:metadata.version(name) for name in ['torch','transformers','peft','accelerate','huggingface-hub','safetensors','tokenizers','flash-linear-attention','fla-core']}
(root/'installation.json').write_text(json.dumps({'schema_version':1,'isolated_venv':True,'python':sys.version.split()[0],'core_packages':versions},indent=2)+'\n')
print('Isolated 27B environment installed; run doctor before training.')
PY
