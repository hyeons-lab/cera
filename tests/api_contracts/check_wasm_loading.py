"""Compare generated CPU WASM loading declarations with the reviewed API shape."""

import argparse
import json
import re
from pathlib import Path

BASELINE = Path(__file__).with_name("wasm_loading.json")


def check(declarations):
    expected = json.loads(BASELINE.read_text())["classes"]
    text = re.sub(r"/\*.*?\*/", "", declarations, flags=re.DOTALL)
    actual = {}
    for match in re.finditer(r"export class (\w+) \{(.*?)\n\}", text, re.DOTALL):
        name, body = match.groups()
        if name not in expected:
            continue
        if name in actual:
            raise ValueError(f"Duplicate loading declaration: {name}")
        actual[name] = "\n".join(
            line.strip() for line in body.splitlines() if line.strip()
        )
    changed = sorted(name for name in expected if actual.get(name) != expected[name])
    if changed:
        raise ValueError(f"Loading declaration drift: {', '.join(changed)}")
    return len(expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("declarations", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            {"status": "passed", "classes": check(args.declarations.read_text())}
        )
    )
