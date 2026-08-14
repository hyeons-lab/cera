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
    model = "gemini-3.7-flash"
    thinking_budget = int(os.getenv("ANTIGRAVITY_THINKING_BUDGET", "8192"))
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 16384,
            "thinkingConfig": {
                "thinkingBudget": thinking_budget,
            },
        },
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text_parts = [
                    p.get("text", "")
                    for p in parts
                    if "text" in p and not p.get("thought", False)
                ]
                if text_parts:
                    return "".join(text_parts).strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"Gemini API model '{model}' HTTPError {e.code}: {body}", file=sys.stderr)
    except Exception as e:
        print(f"Gemini API model '{model}' request failed: {e}", file=sys.stderr)

    return ""


import datetime


def post_or_update_comment(
    github_token: str,
    repo: str,
    pr_number: str,
    comment_body: str,
    head_sha: str = "",
) -> None:
    short_sha = head_sha[:7] if head_sha else run_cmd(["git", "rev-parse", "--short", "HEAD"])
    commit_link = (
        f"[`{short_sha}`](https://github.com/{repo}/commit/{head_sha})"
        if head_sha
        else f"`{short_sha}`"
    )
    timestamp_utc = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

    body_clean = comment_body.strip()
    if body_clean.startswith("## 🪐 Antigravity Code Review"):
        body_clean = body_clean[len("## 🪐 Antigravity Code Review"):].strip()
    elif body_clean.startswith("# Antigravity Code Review"):
        body_clean = body_clean[len("# Antigravity Code Review"):].strip()

    meta_header = (
        f"> *Reviewed commit {commit_link} • {timestamp_utc} • "
        f"Deep Reasoning Audit (Gemini 3.7 Flash)*"
    )
    full_body = f"{COMMENT_TAG}\n## 🪐 Antigravity Code Review\n{meta_header}\n\n{body_clean}"
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
    head_sha = os.getenv("HEAD_SHA", "")
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

    prompt = f"""You are Antigravity, a Senior Principal Systems & Multiplatform Architect conducting a thorough, rigorous, production-grade code review for the Cera repository.

About Cera:
Cera is a high-performance, lightweight LLM / SLM inference engine written in Rust with bindings for Swift, Kotlin, Dart/Flutter, and WebAssembly / WebGPU.

Repository Guidelines & Rules:
{guidelines}

Pull Request Context:
- Title: {pr_title}
- Description: {pr_body}

Diff to review:
```diff
{diff}
```

Review Objective & Standard:
Provide a rigorous, deep-dive code review that matches or exceeds the depth, precision, and actionable quality of top-tier reviewers (such as Copilot / Junie / human staff engineers). Do NOT write superficial or generic summaries. Every finding must be specific, backed by exact file paths, line references or code snippets, and include a concrete fix proposal.

Audit Checklist:
1. **Architectural & Cross-Platform Integrity**:
   - Clean abstractions, minimal coupling, proper module boundaries.
   - Cross-platform differences (macOS vs iOS vs Android vs Web / WASM, desktop vs mobile).
   - Compatibility across Flutter versions, Swift / Kotlin FFI patterns, and memory models.
2. **Concurrency, Cancellation & Asynchronous Correctness**:
   - Atomic variables, ordering semantics, race conditions on shared state.
   - Cancellation flag handling: pre-armed cancels, in-flight preemption, reset semantics across calls.
   - Stream subscriptions, Completers, async/await suspension leaks or race conditions.
3. **Memory, KV Cache & Resource Management**:
   - Model weights lifecycle, KV cache allocations, unclosed sessions or engines.
   - Buffer clones vs borrows, zero-copy safety, web worker neutering vs double-buffering.
   - Dispose patterns in UI frameworks and RAII cleanup in Rust.
4. **Correctness, Edge Cases & Error Domains**:
   - Out-of-bounds access, division by zero, empty collections, nil/null propagation.
   - Graceful error propagation instead of silent failure or unhandled exceptions/panics.
   - Fallback behaviors (e.g. offline cache miss, missing preferences).
5. **Performance & Hot Paths**:
   - Unnecessary heap allocations, redundant sorting in tight loops, string formatting in generation hot paths.
   - Bilinear interpolation, token decoding throughput, and memory layout.
6. **Test Coverage & Regression Prevention**:
   - Missing unit tests for new code paths, unverified edge cases, untested failure branches.

Output Structure:
### 1. Executive Summary & Impact Analysis
- Concise technical synthesis of what the PR changes, architectural implications, and readiness for merge.

### 2. 🚨 Critical / Blocking Issues
*(If none, explicitly state "None identified.")*
- Severe bugs, memory/concurrency violations, unhandled panics/exceptions, or breaking regressions.
- Include exact `file_path:line_number`, explanation of the hazard, and a concrete before/after code diff fix.

### 3. ⚠️ Warnings & Correctness Risks
- Edge cases, error handling gaps, resource leaks, cross-platform caveats, or subtle logical flaws.
- Include exact `file_path:line_number`, explanation, and actionable code diff recommendations.

### 4. 💡 Suggestions & Optimization Opportunities
- Non-blocking improvements: cleaner idioms, performance optimizations in hot paths, documentation comments, or refactoring opportunities.
- Include exact `file_path:line_number` and concise code proposals.

### 5. 🧪 Test Coverage & Edge Cases to Consider
- Specific scenarios or test cases that should be verified (e.g. cancellation during prefill, network failure during stream, zero-sized inputs).

### 6. 🌟 Architecture Highlights
- Acknowledge well-crafted patterns, elegant abstractions, and robust implementations in the PR.

Be direct, highly technical, actionable, and precise.
"""

    print("Generating thorough deep-reasoning code review with Gemini 3.7 Flash...")
    review = call_gemini_api(api_key, prompt)
    if not review:
        print("Failed to get review from API.", file=sys.stderr)
        sys.exit(1)

    print("Posting review comment to GitHub...")
    post_or_update_comment(github_token, repo, pr_number, review, head_sha=head_sha)


if __name__ == "__main__":
    main()
