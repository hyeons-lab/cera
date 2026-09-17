"""Check reviewed Rust declarations; runtime and target compatibility are separate."""

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).with_name("retained_api.json")
RAW_STRING = re.compile(r'(?:br|cr|r)(#*)"')
STRING = re.compile(r'"(?:\\[\s\S]|[^"\\])*"')
CHAR = re.compile(r"'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F_]+\}|[^\r\n])|[^'\\\r\n])'")


def normalize(text):
    clean, _, literals = lexical_views(text)
    result, start = [], 0
    for first, last in literals:
        result.append(re.sub(r"\s+", " ", clean[start:first]))
        result.append(clean[first:last])
        start = last
    result.append(re.sub(r"\s+", " ", clean[start:]))
    return "".join(result).strip()


def lexical_views(text):
    """Mask comments/literals for scanning, preserving offsets and literal payloads."""
    clean, code = list(text), list(text)
    literals = []

    def mask(start, end, comment=False):
        if not comment:
            literals.append((start, end))
        for index in range(start, end):
            if text[index] not in "\r\n":
                code[index] = " "
                if comment:
                    clean[index] = " "

    pos = 0
    while pos < len(text):
        start = pos
        if text.startswith("//", pos):
            end = text.find("\n", pos)
            pos = len(text) if end < 0 else end
            mask(start, pos, comment=True)
        elif text.startswith("/*", pos):
            pos, depth = pos + 2, 1
            while depth and pos < len(text):
                if text.startswith("/*", pos):
                    depth, pos = depth + 1, pos + 2
                elif text.startswith("*/", pos):
                    depth, pos = depth - 1, pos + 2
                else:
                    pos += 1
            if depth:
                raise ValueError("Unclosed block comment")
            mask(start, pos, comment=True)
        elif raw := RAW_STRING.match(text, pos):
            terminator = '"' + raw[1]
            end = text.find(terminator, raw.end())
            if end < 0:
                raise ValueError("Unclosed raw string")
            pos = end + len(terminator)
            mask(start, pos)
        elif text[pos] == '"':
            string = STRING.match(text, pos)
            if string is None:
                raise ValueError("Unclosed string")
            pos = string.end()
            mask(start, pos)
        elif text[pos] == "'" and (char := CHAR.match(text, pos)):
            # A lifetime such as 'a has no closing quote and stays in the code.
            pos = char.end()
            mask(start, pos)
        else:
            pos += 1
    return "".join(clean), "".join(code), literals


# Cache the large source files without letting individual normalized signatures
# evict them while checking each owner in the same file.
source_views = lru_cache(maxsize=8)(lexical_views)


def blocks(text, pattern):
    """Read rustfmt-style blocks using their opening indentation, not inner braces."""
    clean, code, _ = source_views(text)
    for match in re.finditer(pattern, code, re.MULTILINE):
        end = re.search(
            rf"^{match['indent']}\}}\s*$", code[match.end() :], re.MULTILINE
        )
        if end is None:
            raise ValueError(f"Unclosed declaration: {match[0]}")
        span = slice(match.start(), match.end() + end.end())
        yield match["indent"], clean[span], code[span]


def methods(text, owner):
    result = {}
    for indent, block, code in blocks(
        text, rf"^(?P<indent> *)impl {re.escape(owner)}\s*\{{"
    ):
        prefix = re.escape(indent + "    ")
        for match in re.finditer(
            rf"^{prefix}pub (?:async )?fn (\w+)", code, re.MULTILINE
        ):
            end = code.index("{", match.end())
            signature = normalize(block[match.start() : end])
            if match[1] in result:
                raise ValueError(
                    f"Duplicate method requires explicit review: {owner}::{match[1]}"
                )
            result[match[1]] = signature
        for match in re.finditer(rf"^{prefix}pub const (\w+)[^=]+", code, re.MULTILINE):
            if match[1] in result:
                raise ValueError(f"Duplicate constant: {owner}::{match[1]}")
            result[match[1]] = normalize(block[match.start() : match.end()])
    return dict(sorted(result.items()))


def record(text, name):
    found = list(
        blocks(
            text,
            rf"^(?P<indent> *)pub (?:struct|enum|trait) {re.escape(name)}(?:\s*:[^{{}}]+)?\s*\{{",
        )
    )
    if len(found) != 1:
        raise ValueError(f"Expected one record/enum declaration: {name}")
    return normalize(found[0][1])


def functions(text):
    result = {}
    clean, code, _ = source_views(text)
    for match in re.finditer(r"^pub (?:async )?fn (\w+)", code, re.MULTILINE):
        end = code.index("{", match.end())
        result[match[1]] = normalize(clean[match.start() : end])
    return dict(sorted(result.items()))


def declarations(text, item):
    match item["kind"]:
        case "methods":
            return methods(text, item["owner"])
        case "record" | "trait":
            return record(text, item["owner"])
        case "functions":
            return functions(text)
        case other:
            raise ValueError(f"Unknown inventory kind: {other}")


def check(root, baseline):
    errors = []
    identities = set()
    for item in baseline["surfaces"]:
        key = (item["source"], item["kind"], item["owner"])
        if key in identities:
            raise ValueError(f"Duplicate inventory row: {key}")
        identities.add(key)
        if not item["retained_home"] or not item["expected"]:
            raise ValueError(f"Empty retention row: {key}")
        try:
            actual = declarations((root / item["source"]).read_text(), item)
        except (OSError, ValueError) as error:
            errors.append(f"{item['source']}::{item['owner']}: {error}")
            continue
        expected = item["expected"]
        if actual != expected:
            if isinstance(expected, dict) and isinstance(actual, dict):
                details = {
                    "missing": sorted(expected.keys() - actual.keys()),
                    "added": sorted(actual.keys() - expected.keys()),
                    "changed": sorted(
                        k
                        for k in expected.keys() & actual.keys()
                        if expected[k] != actual[k]
                    ),
                }
            else:
                details = "record fields, variants, attributes or defaults changed"
            errors.append(f"{item['source']}::{item['owner']}: {details}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    baseline = json.loads(BASELINE.read_text())
    errors = check(args.root, baseline)
    report = {
        "status": "failed" if errors else "passed",
        "surfaces": len(baseline["surfaces"]),
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
