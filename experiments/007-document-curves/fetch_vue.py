"""Materialize the frozen Vue Chinese documentation split without shipping upstream text."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
from urllib.request import urlopen

REPO = "vuejs-translations/docs-zh-cn"
REV = "9f4cd576000707fe65ec600fb154724951dcdfd0"
URL = f"https://codeload.github.com/{REPO}/tar.gz/{REV}"
SECTIONS = {"guide", "api"}


def split_archive(archive, out):
    groups = defaultdict(list)
    with tarfile.open(archive) as tar:
        root = f"docs-zh-cn-{REV}/"
        (out / "LICENSE").write_bytes(tar.extractfile(root + "LICENSE").read())
        for member in tar:
            if not member.isfile() or not member.name.startswith(root) or member.size < 512:
                continue
            path = Path(member.name[len(root):])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unexpected archive path")
            if len(path.parts) < 3 or path.parts[0] != "src" or path.parts[1] not in SECTIONS or path.suffix != ".md":
                continue
            groups[path.parts[1]].append((path.as_posix(), member))
        index = []
        for group in sorted(groups):
            pages = sorted(groups[group], key=lambda pair: hashlib.sha256(pair[0].encode()).digest())
            for i, (path, member) in enumerate(pages):
                split = "val" if i % 4 == 0 else "train"
                raw = tar.extractfile(member).read()
                if not any("\u4e00" <= c <= "\u9fff" for c in raw.decode()):
                    raise ValueError(f"Expected Chinese documentation in {path}")
                target = out / split / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                index.append({"path": path, "split": split, "sha256": hashlib.sha256(raw).hexdigest()})
        if len(index) != 80:
            raise ValueError("Unexpected number of source pages")
    (out / "source-files.json").write_text(json.dumps({"repository": REPO, "revision": REV, "pages": index}, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--archive", type=Path, help="use a previously downloaded commit archive")
    args = p.parse_args()
    if args.out.exists():
        p.error("choose a new output directory")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        root = Path(tmp)
        archive = args.archive or root / "vue.tar.gz"
        if args.archive is None:
            with urlopen(URL, timeout=90) as src, archive.open("wb") as dest:
                shutil.copyfileobj(src, dest)
        data = root / "split"
        data.mkdir()
        split_archive(archive, data)
        data.rename(args.out)


if __name__ == "__main__":
    main()
