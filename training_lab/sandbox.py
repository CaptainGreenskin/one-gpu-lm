"""Small Linux-only, offline code evaluator. Not a multi-tenant security boundary."""
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import tempfile


def extract_code(text):
    blocks = re.findall(r"```(?:[a-zA-Z0-9_+-]*)\s*\n(.*?)```", text, re.S)
    return blocks[0].strip() if blocks else text.strip()


class Sandbox:
    def __init__(self, node=None):
        self.node = Path(node).resolve() if node else None

    def execute(self, root, command, stdin="", timeout=10, node=False):
        args = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
                "--cap-drop", "ALL", "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
                "--setenv", "LC_ALL", "C.UTF-8", "--ro-bind", "/usr", "/usr",
                "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
                "--symlink", "usr/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev",
                "--tmpfs", "/tmp", "--bind", str(root), "/work", "--chdir", "/work"]
        if node:
            if not self.node:
                raise ValueError("JavaScript requires an explicit Node binary")
            args += ["--ro-bind", str(self.node), "/node"]
        args += ["--"] + command

        def limits():
            resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout + 1))
            resource.setrlimit(resource.RLIMIT_AS, ((16 if node else 2) * 2**30,) * 2)
            resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 2**20,) * 2)
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
            resource.setrlimit(resource.RLIMIT_NPROC, (128, 128))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        # Capture to bounded regular files, rather than an unbounded PIPE.
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as error:
            proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=output, stderr=error,
                                    preexec_fn=limits, start_new_session=True)
            try:
                proc.communicate(stdin.encode(), timeout=timeout + 2)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                return {"ok": False, "reason": "timeout"}
            output.seek(0)
            raw = output.read(2**20 + 1)
            if len(raw) > 2**20:
                return {"ok": False, "reason": "output_limit"}
            return {"ok": proc.returncode == 0, "reason": "ok" if proc.returncode == 0 else "runtime_error",
                    "stdout": raw.decode(errors="replace")}

    def score(self, task, code):
        language = task["language"]
        with tempfile.TemporaryDirectory(prefix="code-eval-") as temporary:
            root = Path(temporary)
            if language == "python":
                (root / "candidate.py").write_text(code)
                (root / "runner.py").write_text(
                    "import json,sys\nfrom candidate import solve\nprint(json.dumps(solve(json.load(sys.stdin)),ensure_ascii=False))\n")
                command = ["/usr/bin/python3", "-I", "-c",
                           "import sys;sys.path.insert(0,'/work');exec(open('/work/runner.py').read())"]
            elif language == "javascript":
                (root / "candidate.cjs").write_text(code + "\nmodule.exports=solve;\n")
                (root / "runner.cjs").write_text(
                    "const solve=require('./candidate.cjs');const fs=require('fs');console.log(JSON.stringify(solve(JSON.parse(fs.readFileSync(0,'utf8')))));\n")
                command = ["/node", "--jitless", "--max-old-space-size=128", "runner.cjs"]
            elif language == "cpp":
                (root / "candidate.cpp").write_text("#include <bits/stdc++.h>\n" + code +
                    '\nint main(){std::vector<long long> a;long long v;while(std::cin>>v)a.push_back(v);std::cout<<solve(a)<<"\\n";}\n')
                result = self.execute(root, ["/usr/bin/g++", "-std=c++17", "-O0", "candidate.cpp", "-o", "candidate"], timeout=25)
                if not result["ok"]:
                    return {"passed": False, "passed_cases": 0, "cases": len(task["cases"]), "reason": "compile_error"}
                command = ["/work/candidate"]
            else:
                raise ValueError("Unsupported language")
            passed, reasons = 0, []
            for case in task["cases"]:
                stdin = " ".join(map(str, case["input"])) if language == "cpp" else json.dumps(case["input"])
                result = self.execute(root, command, stdin, node=language == "javascript")
                reason = result["reason"]
                if result["ok"]:
                    try:
                        actual = json.loads(result["stdout"])
                        # JSON type equality matters: true must not count as integer 1.
                        correct = json.dumps(actual, sort_keys=True) == json.dumps(case["expected"], sort_keys=True)
                        reason = "ok" if correct else "wrong_answer"
                    except (ValueError, TypeError):
                        reason = "invalid_output"
                passed += reason == "ok"
                reasons.append(reason)
            return {"passed": passed == len(task["cases"]), "passed_cases": passed,
                    "cases": len(task["cases"]), "reason": next((r for r in reasons if r != "ok"), "ok")}
