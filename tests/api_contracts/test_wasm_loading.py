"""Mutation controls for the generated loading declaration guard."""

import argparse
import json
from pathlib import Path

from check_wasm_loading import check


def controls(declarations):
    assert check(declarations) == 8
    mutations = {
        "method rename": ("buildGenerative():", "build_generative():"),
        "return type": (
            "asGenerative(): GenerativeModel | undefined;",
            "asGenerative(): GenerativeModel;",
        ),
        "optional companion": ("get audio_decoder(): Uint8Array | undefined;", ""),
        "missing class": ("export class ModelLoader {", "export class RemovedLoader {"),
    }
    for label, (before, after) in mutations.items():
        assert declarations.count(before) == 1, label
        try:
            check(declarations.replace(before, after, 1))
        except ValueError as error:
            assert "Loading declaration drift:" in str(error), (label, error)
        else:
            raise AssertionError(f"Undetected declaration mutation: {label}")
    try:
        check(declarations + "\nexport class ModelLoader {\n}\n")
    except ValueError as error:
        assert "Duplicate loading declaration: ModelLoader" in str(error), error
    else:
        raise AssertionError("Undetected duplicate loading class")
    return len(mutations) + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("declarations", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            {"status": "passed", "mutations": controls(args.declarations.read_text())}
        )
    )
