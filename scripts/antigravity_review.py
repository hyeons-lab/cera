#!/usr/bin/env python3
"""
Antigravity / Gemini automated code review for GitHub Pull Requests.

Fetches the pull request diff, ignores generated files and build artifacts,
applies repository guidelines from AGENTS.md / GEMINI.md, and posts or updates
a structured review comment on the PR.
"""

import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

# Files to exclude from LLM diff review to save context and avoid reviewing generated code
IGNORED_PATTERNS = [
    "cera-ffi/apple/Sources/Cera/cera_ffi.swift",
    "cera-ffi/bindings/kotlin/uniffi/cera_ffi/cera_ffi.kt",
    "cera-ffi/bindings/swift/CeraFFI.h",
    "cera-ffi/bindings/swift/cera_ffi.swift",
    "cera_ffi/lib/src/generated/",
    "Cargo.lock",
    "pubspec.lock",
    "package-lock.json",
    "third_party/",
    "target/",
    ".wasm",
    ".dylib",
    ".so",
    ".dll",
]

COMMENT_TAG = "<!-- antigravity-code-review -->"


def run_cmd(cmd: list[str]) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Command failed: {' '.join(cmd)}\n{res.stderr}", file=sys.stderr)
        return ""
    return res.stdout.strip()


def should_ignore_file(filepath: str) -> bool:
    for pat in IGNORED_PATTERNS:
        if pat.endswith("/"):
            if pat in filepath:
                return True
        elif filepath.endswith(pat) or pat in filepath:
            return True
    return False


def get_pr_diff(base_ref: str) -> str:
    # Ensure origin/base_ref is fetched
    run_cmd(["git", "fetch", "origin", base_ref])

    changed_files = run_cmd(["git", "diff", "--name-only", f"origin/{base_ref}...HEAD"]).splitlines()
    reviewable_files = [f for f in changed_files if not should_ignore_file(f)]

    if not reviewable_files:
        return ""

    diff_chunks = []
    for f in reviewable_files:
        diff = run_cmd(["git", "diff", f"origin/{base_ref}...HEAD", "--", f])
        if diff:
            # Cap individual large file diffs if needed
            if len(diff) > 20000:
                diff = diff[:20000] + "\n\n[... diff truncated for length ...]\n"
            diff_chunks.append(diff)

    full_diff = "\n\n".join(diff_chunks)
    # Global safety cap (approx 80k characters)
    if len(full_diff) > 80000:
        full_diff = full_diff[:80000] + "\n\n[... overall diff truncated for context limit ...]\n"
    return full_diff


def get_repo_guidelines() -> str:
    guidelines = []
    for filename in ["AGENTS.md", "GEMINI.md"]:
        if os.path.isfile(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    guidelines.append(f"--- {filename} ---\n" + f.read(4000))
            except Exception:
                pass
    return "\n\n".join(guidelines)


def call_gemini_api(api_key: str, prompt: str) -> str:
    candidate_models = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 3000,
        },
    }
    data_bytes = json.dumps(payload).encode("utf-8")

    for model in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            print(f"Gemini API model '{model}' HTTPError {e.code}: {body}", file=sys.stderr)
            if e.code == 404:
                continue
        except Exception as e:
            print(f"Gemini API model '{model}' request failed: {e}", file=sys.stderr)

    return ""


def post_or_update_comment(github_token: str, repo: str, pr_number: str, comment_body: str) -> None:
    full_body = f"{COMMENT_TAG}\n## 🪐 Antigravity Code Review\n\n{comment_body}"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Antigravity-Code-Review",
    }

    # Check for existing comment with COMMENT_TAG
    list_url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    req = urllib.request.Request(list_url, headers=headers)
    existing_comment_id = None

    try:
        with urllib.request.urlopen(req) as resp:
            comments = json.loads(resp.read().decode("utf-8"))
            for c in comments:
                if COMMENT_TAG in c.get("body", ""):
                    existing_comment_id = c.get("id")
                    break
    except Exception as e:
        print(f"Error fetching existing comments: {e}", file=sys.stderr)

    if existing_comment_id:
        # Update existing comment
        update_url = f"https://api.github.com/repos/{repo}/issues/comments/{existing_comment_id}"
        req = urllib.request.Request(
            update_url,
            data=json.dumps({"body": full_body}).encode("utf-8"),
            headers=headers,
            method="PATCH",
        )
        try:
            with urllib.request.urlopen(req):
                print(f"Updated existing review comment #{existing_comment_id}")
        except Exception as e:
            print(f"Error updating comment: {e}", file=sys.stderr)
    else:
        # Create new comment
        create_url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
        req = urllib.request.Request(
            create_url,
            data=json.dumps({"body": full_body}).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req):
                print("Created new review comment on PR")
        except Exception as e:
            print(f"Error posting comment: {e}", file=sys.stderr)


def main():
    api_key = os.getenv("ANTIGRAVITY_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ANTIGRAVITY_API_KEY or GEMINI_API_KEY not set; skipping automated code review.", file=sys.stderr)
        sys.exit(0)

    github_token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")
    pr_number = os.getenv("PR_NUMBER")
    base_ref = os.getenv("BASE_REF", "main")
    pr_title = os.getenv("PR_TITLE", "")
    pr_body = os.getenv("PR_BODY", "")

    if not (github_token and repo and pr_number):
        print("Missing GitHub environment context (GITHUB_TOKEN, GITHUB_REPOSITORY, PR_NUMBER).", file=sys.stderr)
        sys.exit(0)

    diff = get_pr_diff(base_ref)
    if not diff:
        print("No reviewable diff found.")
        sys.exit(0)

    guidelines = get_repo_guidelines()

    prompt = f"""You are Antigravity, an expert AI code reviewer evaluating a pull request for the Cera repository.
Cera is a lightweight inference engine and library for LLM / SLM (written in Rust with FFI bindings for Swift, Kotlin, Dart/Flutter, and WebAssembly).

Repository Guidelines & Conventions:
{guidelines}

Pull Request Context:
Title: {pr_title}
Description: {pr_body}

Diff to review:
```diff
{diff}
```

Review Instructions:
1. Provide a concise, high-level summary of what this PR does.
2. Review the code changes for:
   - Correctness, concurrency/lifetime safety, resource leaks (e.g. KV cache, model weights, subscriptions).
   - Invariants and edge case handling (e.g. error propagation, bounds, null checks).
   - Performance and unnecessary allocations in hot paths.
   - Adherence to project conventions (clean API surfaces, backward/forward compatibility, proper derives and error types).
3. Clearly organize findings by severity:
   - 🚨 **Critical / Blocking**: Serious bugs, memory/concurrency violations, breaking API regressions.
   - ⚠️ **Warnings / Suggestions**: Edge case risks, performance improvements, cleaner idioms.
   - ✅ **Highlights**: Well-designed aspects of the change.
4. If everything looks clean, state that explicitly.
5. Be actionable, concise, and constructive.
"""

    print("Generating code review with Gemini / Antigravity...")
    review = call_gemini_api(api_key, prompt)
    if not review:
        print("Failed to get review from API.", file=sys.stderr)
        sys.exit(1)

    print("Posting review comment to GitHub...")
    post_or_update_comment(github_token, repo, pr_number, review)


if __name__ == "__main__":
    main()
