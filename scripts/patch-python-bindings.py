"""Install the handwritten streaming extension on the generated module entry point."""

from pathlib import Path

MARKER = "# Install cera streaming helpers (scripts/patch-python-bindings.py)."
FOOTER = '''
# Install cera streaming helpers (scripts/patch-python-bindings.py).
if __package__:
    from .cera_ffi_streaming import install as _install_streaming
else:
    from cera_ffi_streaming import install as _install_streaming
_install_streaming(sys.modules[__name__])
del _install_streaming
'''


def main():
    path = Path(__file__).resolve().parents[1] / "cera-ffi/bindings/python/cera_ffi.py"
    source = path.read_text().split(MARKER, 1)[0].rstrip()
    path.write_text(source + "\n" + FOOTER)


if __name__ == "__main__":
    main()
