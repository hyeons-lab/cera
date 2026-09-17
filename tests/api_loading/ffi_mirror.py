"""Stage production FFI unchanged, then append test observations."""

import shutil

from prepare import REPO, binding_inventory, digest


def prepare_ffi(workspace):
    root = REPO / "cera-ffi"
    target = workspace / "cera-ffi"
    target.mkdir()
    shutil.copyfile(root / "Cargo.toml", target / "Cargo.toml")
    shutil.copytree(root / "src", target / "src")
    sources = binding_inventory(target)
    if sources != binding_inventory(root):
        raise RuntimeError("Production FFI source changed while copying")
    lib = target / "src/lib.rs"
    text = lib.read_text()
    if text.count("    inner: Arc<cera::CeraEngine>,") != 1:
        raise ValueError("Production shared engine storage changed")
    text += (
        "\n" + (REPO / "tests/api_loading/native/session_observation.rs").read_text()
    )
    lib.write_text(text)
    session = workspace / "cera/src/session.rs"
    session.write_text(
        session.read_text() + "\nimpl Session {\n"
        "    pub fn config_for_loading_probe(&self) -> SessionConfig { self.config.clone() }\n"
        "}\n"
    )
    return sources, {str(p.relative_to(workspace)): digest(p) for p in (lib, session)}
