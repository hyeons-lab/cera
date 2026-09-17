"""Stage production CPU WASM unchanged, then append test observations."""

import shutil

from prepare import REPO, binding_inventory, digest


def prepare_wasm(workspace):
    root = REPO / "cera-wasm"
    target = workspace / "cera-wasm"
    target.mkdir()
    shutil.copyfile(root / "Cargo.toml", target / "Cargo.toml")
    shutil.copytree(root / "src", target / "src")
    sources = binding_inventory(target)
    if sources != binding_inventory(root):
        raise RuntimeError("Production WASM source changed while copying")
    changes = {}
    for name, observer in (
        ("lib.rs", "session_observation.rs"),
        ("loading.rs", "loading_observation.rs"),
    ):
        path = target / "src" / name
        path.write_text(
            path.read_text()
            + "\n"
            + (REPO / "tests/api_loading/wasm" / observer).read_text()
        )
        changes[str(path.relative_to(workspace))] = digest(path)
    return sources, changes
