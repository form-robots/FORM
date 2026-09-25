"""Verify, bundle, and install the paper's recorded inputs and reference outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data/manifest.json'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def records(bundle=None):
    rows = json.loads(MANIFEST.read_text())['files']
    return [r for r in rows if bundle is None or r['bundle'] == bundle]


def verify(bundle=None):
    errors = []
    rows = records(bundle)
    for row in rows:
        path = ROOT / row['path']
        if not path.is_file():
            errors.append(f"Missing: {row['path']}")
        elif path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            errors.append(f"Checksum mismatch: {row['path']}")
    if errors:
        raise SystemExit('\n'.join(errors[:30]) + f'\n{len(errors)} invalid files')
    print(f'Verified {len(rows)} files ({sum(r["bytes"] for r in rows) / 1e9:.2f} GB).')


def mount():
    """Restore historical relative paths without links to another checkout."""
    for storage, prefix in [('data/artifacts', 'out'), ('data/raw', '')]:
        directory = ROOT / storage
        if not directory.exists():
            continue
        for child in directory.iterdir():
            link = (ROOT / 'presentation/assets' if child.name == 'presentation_assets'
                    else ROOT / prefix / child.name)
            if link.is_symlink():
                if link.resolve() != child.resolve():
                    raise RuntimeError(f'Conflicting mount: {link}')
            elif link.exists():
                raise RuntimeError(f'Refusing to replace existing directory: {link}')
            else:
                import os
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(os.path.relpath(child, link.parent), target_is_directory=True)


def pack(destination):
    destination.mkdir(parents=True, exist_ok=True)
    index = []
    for bundle in sorted({r['bundle'] for r in records()}):
        target = destination / f'form-{bundle}.tar.gz'
        if target.exists():
            raise FileExistsError(target)
        verify(bundle)
        with tarfile.open(target, 'w:gz', compresslevel=1) as archive:
            for row in records(bundle):
                archive.add(ROOT / row['path'], arcname=row['path'], recursive=False)
        index.append(dict(file=target.name, sha256=digest(target), bytes=target.stat().st_size))
        print(f'Packed {target.name}', flush=True)
    (destination / 'bundles.json').write_text(json.dumps(index, indent=2) + '\n')


def unpack(paths):
    allowed = {r['path']: r for r in records()}
    for path in paths:
        with tarfile.open(path, 'r:*') as archive:
            for member in archive:
                if member.name not in allowed or not member.isfile():
                    raise ValueError(f'Unexpected archive member: {member.name}')
                target = ROOT / member.name
                if target.exists():
                    if digest(target) != allowed[member.name]['sha256']:
                        raise FileExistsError(f'Conflicting local data: {target}')
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                with archive.extractfile(member) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output)
                if digest(target) != allowed[member.name]['sha256']:
                    target.unlink()
                    raise ValueError(f'Corrupt archive member: {member.name}')
    mount()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'mount', 'pack', 'unpack'])
    parser.add_argument('paths', nargs='*', type=Path)
    parser.add_argument('--bundle')
    args = parser.parse_args()
    if args.command == 'verify':
        verify(args.bundle)
    elif args.command == 'mount':
        mount()
    elif args.command == 'pack':
        pack(args.paths[0] if args.paths else ROOT / 'data/bundles')
    else:
        unpack(args.paths)
