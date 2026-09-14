"""Stage production core and candidate bindings in an isolated workspace."""

import hashlib
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root, excluded=()):
    return {
        str(p.relative_to(root)): digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and not set(p.relative_to(root).parts).intersection(excluded)
    }


def verify_inventory(root, expected, excluded=()):
    if inventory(root, excluded) != expected:
        raise RuntimeError(f"Input/artifact drift: {root}")


def binding_inventory(root):
    return {
        f"{root.name}/Cargo.toml": digest(root / "Cargo.toml"),
        **{f"{root.name}/src/{p}": h for p, h in inventory(root / "src").items()},
    }


def cargo_configs(workspace, cargo_home):
    """Include absent files so a newly introduced ancestor config is also drift."""
    directories = [p / ".cargo" for p in (workspace, *workspace.parents)]
    directories.append(cargo_home)
    return {
        str(p): digest(p) if p.is_file() else None
        for directory in directories
        for name in ("config", "config.toml")
        for p in [directory / name]
    }


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Expected one exact adaptation anchor: {before!r}")
    return text.replace(before, after, 1)


def prepare(output):
    workspace = output / "workspace"
    workspace.mkdir()
    manifest = (REPO / "Cargo.toml").read_text()
    manifest, count = re.subn(
        r"^members = .*",
        'members = ["cera", "cera-ffi", "cera-wasm", "native", "wasm", "consumer"]',
        manifest,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise ValueError("Workspace member anchor changed")
    (workspace / "Cargo.toml").write_text(manifest)
    shutil.copyfile(REPO / "Cargo.lock", workspace / "Cargo.lock")
    original = {}
    for source in sorted((REPO / "cera").rglob("*")):
        if not source.is_file() or "target" in source.parts:
            continue
        relative = source.relative_to(REPO)
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        original[str(relative)] = digest(target)
        if digest(source) != original[str(relative)]:
            raise RuntimeError(f"Source changed while copying: {relative}")
    for name in ("native", "wasm", "consumer"):
        shutil.copytree(ROOT / name, workspace / name)
    shutil.copyfile(ROOT / "shared.rs", workspace / "shared.rs")
    shutil.copyfile(ROOT / "defaults.rs", workspace / "defaults.rs")
    shutil.copyfile(ROOT / "context.rs", workspace / "context.rs")
    changes = {}
    from ffi_mirror import prepare_ffi

    ffi_sources, ffi_changes = prepare_ffi(workspace)
    changes.update(ffi_changes)
    from wasm_mirror import prepare_wasm

    wasm_sources, wasm_changes = prepare_wasm(workspace)
    changes.update(wasm_changes)
    return workspace, {
        "source_sha256": original,
        "ffi_source_sha256": ffi_sources,
        "wasm_source_sha256": wasm_sources,
        "exposed_sha256": changes,
        "workspace_inputs_sha256": {
            name: digest(REPO / name) for name in ("Cargo.toml", "Cargo.lock")
        },
    }
