"""Screen the agent-visible mount namespace for known package/reference copies.

This is a bounded inventory check, not a proof about pretraining or all possible
renamed/embedded implementations. Run inside the same sandbox used by the agent.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile


def scan(roots, package, canonical, target_hashes):
    canonical = Path(canonical).resolve()
    if not canonical.is_dir() or not roots or any(not Path(root).is_dir() for root in roots):
        raise ValueError('Every requested scan root and the canonical source must exist')
    if not package or not target_hashes or any(len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in target_hashes):
        raise ValueError('Require package name and nonempty valid source hashes')
    named = set()
    duplicates = set()
    archives = set()
    unreadable = 0
    unreadable_details = []
    files = 0
    seen_directories = set()

    def walk_error(error):
        nonlocal unreadable
        unreadable += 1
        unreadable_details.append({'path': str(error.filename), 'error_type': type(error).__name__, 'errno': error.errno})

    for root in roots:
        for directory, dirs, names in os.walk(root, followlinks=True, onerror=walk_error):
            path = Path(directory)
            try:
                stat = path.stat()
            except OSError as error:
                unreadable += 1
                unreadable_details.append({'path': path.as_posix(), 'error_type': type(error).__name__, 'errno': error.errno})
                dirs[:] = []
                continue
            key = (stat.st_dev, stat.st_ino)
            if key in seen_directories:
                dirs[:] = []
                continue
            seen_directories.add(key)
            inside = path.resolve().is_relative_to(canonical)
            if not inside and package.lower() in path.name.lower():
                named.add(path.as_posix())
            for name in names:
                item = path / name
                files += 1
                if inside:
                    continue
                if package.lower() in name.lower():
                    named.add(item.as_posix())
                try:
                    if item.suffix in {'.py', '.pyi'} and hashlib.sha256(item.read_bytes()).hexdigest() in target_hashes:
                        duplicates.add(item.as_posix())
                    if item.suffix in {'.zip', '.whl'} and zipfile.is_zipfile(item):
                        with zipfile.ZipFile(item) as archive:
                            if any(package.lower() in member.lower() for member in archive.namelist()):
                                archives.add(item.as_posix())
                except (OSError, zipfile.BadZipFile) as error:
                    unreadable += 1
                    unreadable_details.append({'path': item.as_posix(), 'error_type': type(error).__name__, 'errno': getattr(error, 'errno', None)})
    # These checks execute as the agent's sandbox UID, with read-only dependency
    # mounts. ENOENT and EACCES identify inaccessible bytes, not an uninspected
    # readable source. Retain counts and hashes without publishing host filenames.
    blocking_errors = [r for r in unreadable_details if r['errno'] not in {2, 13}]
    safe_details = [{'path_sha256': hashlib.sha256(r['path'].encode()).hexdigest(),
                     'error_type': r['error_type'], 'errno': r['errno']} for r in unreadable_details]
    return {'schema_version': 1, 'roots': list(roots), 'package': package,
            'canonical_source': canonical.as_posix(), 'files_seen': files,
            'directories_seen': len(seen_directories), 'unreadable_entries': unreadable,
            'unreadable_details': safe_details,
            'blocking_scan_errors': len(blocking_errors),
            'other_package_name_matches': sorted(named),
            'duplicate_selected_source_files': sorted(duplicates),
            'archives_containing_package_name': sorted(archives),
            'screen_passed': not (named or duplicates or archives or blocking_errors),
            'readability_limits': 'ENOENT/EACCES entries cannot be read by this same sandbox UID through the read-only mounts; recorded separately rather than called scanned. Other scan errors block admission.',
            'scope': 'Known package names, exact selected .py/.pyi source hashes, and zip/wheel member names in agent-visible roots; not all semantic copies or pretraining exposure.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', required=True)
    p.add_argument('--canonical', required=True)
    p.add_argument('--hashes', required=True, help='JSON array of selected source SHA256 digests')
    p.add_argument('--roots', nargs='+', default=['/usr', '/opt/python-site', '/opt/node', '/opt/repo-harness', '/work'])
    a = p.parse_args()
    print(json.dumps(scan(a.roots, a.package, a.canonical, set(json.loads(a.hashes)))))
