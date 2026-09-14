"""Run native consumers against bounded loopback downloads and isolated stores."""

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from prepare import digest, inventory

COMMIT = "2" * 40
PROFILES = ("hf", "bundle", "manifest", "directory")


def run_remote(commands, language, command, model):
    root = commands.output / f"remote-{language}"
    root.mkdir()
    weights = model.read_bytes().ljust(600 * 1024, b"\0")
    routes = {}
    metadata = {
        "id": "fixture/text",
        "sha": COMMIT,
        "siblings": [
            {"rfilename": "model-Q8_0.gguf"},
            {"rfilename": "model-Q4_K_M.gguf"},
        ],
    }
    routes["/api/models/fixture/text/revision/release"] = json.dumps(metadata).encode()
    routes[f"/fixture/text/resolve/{COMMIT}/model-Q8_0.gguf"] = weights
    routes[f"/fixture/text/resolve/{COMMIT}/model-Q4_K_M.gguf"] = (
        commands.output / "kws.gguf"
    ).read_bytes()
    for name, fixture in (("kind", "kws"), ("assembly", "llama")):
        routes[f"/api/models/fixture/{name}/revision/{COMMIT}"] = json.dumps(
            {
                "id": f"fixture/{name}",
                "sha": COMMIT,
                "siblings": [{"rfilename": "model.gguf"}],
            }
        ).encode()
        routes[f"/fixture/{name}/resolve/{COMMIT}/model.gguf"] = (
            commands.output / f"{fixture}.gguf"
        ).read_bytes()
    for profile in PROFILES:
        routes[f"/assets/{profile}.gguf"] = weights
        routes[f"/after/{profile}.bin"] = b"retained-callback" * 32
    requests = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_CONNECT(self):
            with lock:
                requests.append(["CONNECT", self.path])
            self.send_error(403)

        def do_HEAD(self):
            self.respond(False)

        def do_GET(self):
            self.respond(True)

        def respond(self, body):
            with lock:
                requests.append([self.command, self.path])
            # Proxy requests never forward to public endpoints.
            if urlsplit(self.path).scheme or self.path not in routes:
                self.send_error(404)
                return
            data = routes[self.path]
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("ETag", hashlib.sha256(data).hexdigest())
            self.send_header("Connection", "close")
            self.end_headers()
            if body:
                self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    endpoint = f"http://127.0.0.1:{server.server_port}"
    for profile in PROFILES:
        store = root / profile / "store"
        store.mkdir(parents=True)
        if profile == "bundle":
            path = (
                store / "huggingface.co/LiquidAI/LeapBundles/resolve/main/fixture-model"
            )
            path.mkdir(parents=True)
            (path / "relative.gguf").write_bytes(weights)
            manifest_path = path / "Q8_0.json"
            primary = "relative.gguf"
        else:
            inputs = root / profile / "inputs"
            inputs.mkdir()
            manifest_path = inputs / "model.json"
            primary = f"{endpoint}/assets/{profile}.gguf"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "inference_type": "llama.cpp/text-to-text",
                    "load_time_parameters": {"model": primary},
                    "chat_template": "remote-template",
                }
            )
        )
    (root / "endpoint.txt").write_text(endpoint)
    inputs = inventory(root)
    original = commands.environment
    commands.environment = dict(original)
    commands.environment.update(
        HF_ENDPOINT=endpoint,
        HF_TOKEN="cera-loopback-fixture",
        HF_HOME=str(root / "hf-home"),
        NO_PROXY="127.0.0.1",
        no_proxy="127.0.0.1",
    )
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        commands.environment[name] = endpoint
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = commands.run(f"{language}-run", [*command, model.parent, root])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        commands.environment = original
        (root / "requests.json").write_text(json.dumps(requests, indent=2) + "\n")
    for path, expected in inputs.items():
        if digest(root / path) != expected:
            raise RuntimeError(f"Remote input changed: {path}")
    required = [f"/fixture/text/resolve/{COMMIT}/model-Q8_0.gguf"]
    required += [f"/assets/{p}.gguf" for p in ("manifest", "directory")]
    required += [f"/after/{p}.bin" for p in PROFILES]
    for path in required:
        if requests.count(["GET", path]) != 1:
            raise RuntimeError(
                f"Expected one cold download and silent cache reuse: {path}"
            )
    if ["GET", f"/fixture/text/resolve/{COMMIT}/model-Q4_K_M.gguf"] in requests:
        raise RuntimeError("Explicit quant override was lost")
    report = {
        "status": "passed",
        "inputs_sha256": inputs,
        "routes_sha256": {p: hashlib.sha256(b).hexdigest() for p, b in routes.items()},
        "requests": requests,
        "requests_sha256": digest(root / "requests.json"),
        "required_single_downloads": required,
    }
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return result, {"root": str(root), **report}
