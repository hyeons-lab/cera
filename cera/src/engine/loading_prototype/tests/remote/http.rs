//! Loopback server plus subprocess isolation for process-global HF settings.

use super::*;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

pub(super) const MAIN_COMMIT: &str = "1111111111111111111111111111111111111111";
pub(super) const RELEASE_COMMIT: &str = "2222222222222222222222222222222222222222";

#[derive(Clone)]
pub(super) struct Response {
    status: u16,
    ranges: bool,
    bytes: Vec<u8>,
    pub(super) hash: Option<String>,
    sequence: Option<Arc<Mutex<std::collections::VecDeque<Response>>>>,
    required_auth: Option<String>,
}

impl Response {
    pub(super) fn bytes(bytes: impl AsRef<[u8]>) -> Self {
        Self {
            status: 200,
            ranges: false,
            bytes: bytes.as_ref().to_vec(),
            hash: None,
            sequence: None,
            required_auth: None,
        }
    }

    pub(super) fn sequence(responses: Vec<Self>) -> Self {
        assert!(!responses.is_empty());
        Self {
            sequence: Some(Arc::new(Mutex::new(responses.into()))),
            ..Self::bytes("")
        }
    }

    pub(super) fn status(status: u16) -> Self {
        Self {
            status,
            ..Self::bytes("")
        }
    }

    pub(super) fn authenticated(mut self, token: &str) -> Self {
        self.required_auth = Some(format!("Bearer {token}"));
        self
    }

    fn current(mut self) -> Self {
        if let Some(sequence) = self.sequence.take() {
            let mut queue = sequence.lock().unwrap();
            if queue.len() > 1 {
                queue.pop_front().unwrap()
            } else {
                queue.front().unwrap().clone()
            }
        } else {
            self
        }
    }

    pub(super) fn ranged(bytes: impl AsRef<[u8]>) -> Self {
        Self {
            ranges: true,
            ..Self::bytes(bytes)
        }
    }

    pub(super) fn gguf(bytes: impl AsRef<[u8]>) -> Self {
        Self {
            hash: Some(digest(bytes.as_ref())),
            ..Self::bytes(bytes)
        }
    }
}

pub(super) fn hf_model_bytes() -> Vec<u8> {
    let mut bytes = tiny_llama().to_vec();
    // Unreferenced trailing bytes keep the same valid GGUF and tensors while
    // crossing multiple download-progress thresholds (256 KiB each).
    bytes.resize(600 * 1024, 0);
    bytes
}

fn routes() -> HashMap<String, Response> {
    let mut routes = HashMap::new();
    let metadata = |sha: &str, names: &[&str]| {
        json!({
            "id": "fixture/model", "sha": sha,
        "siblings": names.iter().map(|name| json!({"rfilename": name})).collect::<Vec<_>>()
        })
        .to_string()
    };
    let model = tiny_llama();
    for rev in ["main", "release"] {
        let commit = if rev == "main" {
            MAIN_COMMIT
        } else {
            RELEASE_COMMIT
        };
        let suffix = if rev == "main" {
            ""
        } else {
            "/revision/release"
        };
        routes.insert(
            format!("/api/models/fixture/text{suffix}"),
            Response::bytes(metadata(
                commit,
                &[
                    "model-Q8_0.gguf",
                    "nested/model-Q4_K_M.gguf",
                    "generation_config.json",
                ],
            )),
        );
        for filename in ["model-Q8_0.gguf", "nested/model-Q4_K_M.gguf"] {
            routes.insert(
                format!("/fixture/text/resolve/{commit}/{filename}"),
                Response::gguf(hf_model_bytes()),
            );
        }
        routes.insert(format!("/fixture/text/resolve/{commit}/generation_config.json"),
            Response::bytes(json!({"temperature":0.25,"min_p":0.15,"top_p":0.8,"top_k":3,"repetition_penalty":1.2}).to_string()));
    }
    for arch in [
        "bert",
        "modernbert",
        "whisper",
        "silero_vad",
        "kws",
        "future",
    ] {
        routes.insert(
            format!("/api/models/fixture/{arch}"),
            Response::bytes(metadata(MAIN_COMMIT, &["model-F16.gguf"])),
        );
        routes.insert(
            format!("/fixture/{arch}/resolve/{MAIN_COMMIT}/model-F16.gguf"),
            Response::gguf(header(arch)),
        );
    }
    routes.insert(
        "/api/models/fixture/safetensors".into(),
        Response::bytes(metadata(MAIN_COMMIT, &["model.safetensors"])),
    );
    routes.insert(
        "/api/models/fixture/bad-json".into(),
        Response::bytes("invalid json"),
    );
    routes.insert(
        "/api/models/fixture/private".into(),
        Response {
            status: 401,
            ..Response::bytes("")
        },
    );
    for name in [
        "primary",
        "mmproj",
        "decoder",
        "tokenizer",
        "draft",
        "extra",
        "stale",
        "offline",
    ] {
        routes.insert(format!("/assets/{name}.gguf"), Response::gguf(&model));
    }
    routes.insert("/assets/bad.gguf".into(), Response::gguf("not a gguf"));
    routes.insert(
        "/assets/hash-mismatch.gguf".into(),
        Response {
            hash: Some("0".repeat(64)),
            ..Response::bytes(&model)
        },
    );
    routes.insert("/LiquidAI/LFM2.5-1.2B-Instruct-DSpark-GGUF/resolve/main/LFM2.5-1.2B-Instruct-DSpark-Q4_K_M.gguf".into(), Response::gguf(&model));
    routes
}

struct Server {
    url: String,
    stop: Arc<AtomicBool>,
    requests: Arc<Mutex<Vec<(String, String)>>>,
    ranges: Arc<Mutex<Vec<(String, String)>>>,
    thread: Option<JoinHandle<()>>,
}

impl Server {
    fn start(routes: HashMap<String, Response>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let thread_stop = stop.clone();
        let log = requests.clone();
        let ranges = Arc::new(Mutex::new(Vec::new()));
        let range_log = ranges.clone();
        let thread = thread::spawn(move || {
            while !thread_stop.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((stream, _)) => serve(stream, &routes, &log, &range_log),
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5))
                    }
                    Err(e) => panic!("loopback accept: {e}"),
                }
            }
        });
        Self {
            url,
            stop,
            requests,
            ranges,
            thread: Some(thread),
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let result = thread.join();
            // Preserve the original failure if cleanup runs while unwinding.
            if !thread::panicking() {
                result.expect("loopback server failed");
            }
        }
    }
}

fn serve(
    mut stream: TcpStream,
    routes: &HashMap<String, Response>,
    log: &Mutex<Vec<(String, String)>>,
    range_log: &Mutex<Vec<(String, String)>>,
) {
    // Accepted sockets can inherit the listener's nonblocking mode on macOS.
    // Only accept is polled; each bounded request/response uses blocking I/O.
    stream.set_nonblocking(false).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let mut reader = BufReader::new(&mut stream);
    let mut first = String::new();
    if reader.read_line(&mut first).unwrap_or(0) == 0 {
        return;
    }
    let mut words = first.split_whitespace();
    let method = words.next().unwrap().to_owned();
    let target = words.next().unwrap().to_owned();
    let mut size = first.len();
    let mut range = None;
    let mut authorization = None;
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).unwrap();
        size += n;
        assert!(size < 16 * 1024, "oversized fixture request");
        if let Some((key, value)) = line.split_once(':')
            && key.eq_ignore_ascii_case("range")
        {
            range = Some(value.trim().to_owned());
        }
        if let Some((key, value)) = line.split_once(':')
            && key.eq_ignore_ascii_case("authorization")
        {
            authorization = Some(value.trim().to_owned());
        }
        if n == 0 || line == "\r\n" {
            break;
        }
    }
    log.lock().unwrap().push((method.clone(), target.clone()));
    if let Some(value) = &range {
        range_log
            .lock()
            .unwrap()
            .push((target.clone(), value.clone()));
    }
    // This proxy never forwards traffic. Fixed public bundle URLs can only
    // exercise the preseeded cache's existing failed-HEAD fallback.
    let response = if method == "CONNECT" || (method == "HEAD" && target == "/assets/offline.gguf")
    {
        Response {
            status: 403,
            ..Response::bytes("")
        }
    } else {
        routes.get(&target).cloned().unwrap_or(Response {
            status: 404,
            ..Response::bytes("")
        })
    };
    let response = response.current();
    let response = if response.required_auth.is_some() && response.required_auth != authorization {
        Response::status(403)
    } else {
        response
    };
    let mut status = response.status;
    let mut bytes = response.bytes.as_slice();
    let mut content_range = String::new();
    if response.ranges
        && status == 200
        && method == "GET"
        && let Some(range) = range
    {
        let (start, end) = range
            .strip_prefix("bytes=")
            .unwrap()
            .split_once('-')
            .unwrap();
        let start: usize = start.parse().unwrap();
        let end: usize = end.parse().unwrap();
        if start <= end && end < bytes.len() {
            content_range = format!("Content-Range: bytes {start}-{end}/{}\r\n", bytes.len());
            bytes = &bytes[start..=end];
            status = 206;
        } else {
            status = 416;
            content_range = format!("Content-Range: bytes */{}\r\n", bytes.len());
            bytes = &[];
        }
    }
    let hash = response
        .hash
        .map(|hash| format!("X-Linked-Etag: \"sha256:{hash}\"\r\n"))
        .unwrap_or_default();
    let result = write!(
        stream,
        "HTTP/1.1 {} Fixture\r\nContent-Length: {}\r\n{hash}{content_range}Connection: close\r\n\r\n",
        status,
        bytes.len()
    )
    .and_then(|()| {
        if method != "HEAD" {
            stream.write_all(bytes)
        } else {
            Ok(())
        }
    });
    if let Err(error) = result {
        // Clients may close as soon as they see a rejected CONNECT or headers.
        assert!(
            matches!(
                error.kind(),
                io::ErrorKind::BrokenPipe | io::ErrorKind::ConnectionReset
            ),
            "loopback response: {error}"
        );
    }
}

pub(super) struct Context {
    pub url: String,
    pub root: PathBuf,
}

/// Each invocation runs only its own test in a child whose endpoint, proxy and
/// synthetic token are set before Rust/HTTP worker threads start.
pub(super) fn isolated(
    name: &str,
    test: impl FnOnce(Context),
    check_requests: impl FnOnce(&[(String, String)]),
) {
    isolated_with_routes(name, routes, test, check_requests);
}

pub(super) fn isolated_with_routes(
    name: &str,
    routes: impl FnOnce() -> HashMap<String, Response>,
    test: impl FnOnce(Context),
    check_requests: impl FnOnce(&[(String, String)]),
) {
    isolated_with_ranges(name, routes, test, |requests, _ranges| {
        check_requests(requests)
    });
}

pub(super) fn isolated_with_ranges(
    name: &str,
    routes: impl FnOnce() -> HashMap<String, Response>,
    test: impl FnOnce(Context),
    check_requests: impl FnOnce(&[(String, String)], &[(String, String)]),
) {
    isolated_with_ranges_at(name, "", routes, test, check_requests);
}

pub(super) fn isolated_with_endpoint_path(
    name: &str,
    endpoint_path: &str,
    routes: impl FnOnce() -> HashMap<String, Response>,
    test: impl FnOnce(Context),
    check_requests: impl FnOnce(&[(String, String)]),
) {
    isolated_with_ranges_at(name, endpoint_path, routes, test, |requests, _ranges| {
        check_requests(requests)
    });
}

fn isolated_with_ranges_at(
    name: &str,
    endpoint_path: &str,
    routes: impl FnOnce() -> HashMap<String, Response>,
    test: impl FnOnce(Context),
    check_requests: impl FnOnce(&[(String, String)], &[(String, String)]),
) {
    const CHILD: &str = "CERA_REMOTE_CONTRACT_CHILD";
    if std::env::var(CHILD).as_deref() == Ok(name) {
        test(Context {
            url: std::env::var("HF_ENDPOINT").unwrap(),
            root: PathBuf::from(std::env::var_os("CERA_REMOTE_CONTRACT_ROOT").unwrap()),
        });
        return;
    }
    let server = Server::start(routes());
    let root = tempfile::tempdir().unwrap();
    let output_path = root.path().join("child-output.txt");
    let output = fs::File::create(&output_path).unwrap();
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            &format!(
                "{}::{name}",
                module_path!()
                    .split_once("::")
                    .unwrap()
                    .1
                    .trim_end_matches("::http")
            ),
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD, name)
        .env("CERA_REMOTE_CONTRACT_ROOT", root.path())
        .env("HF_ENDPOINT", format!("{}{endpoint_path}", server.url))
        .env("HF_TOKEN", "cera-loopback-fixture")
        .env("NO_PROXY", "127.0.0.1")
        .env("no_proxy", "127.0.0.1")
        .stdout(Stdio::from(output.try_clone().unwrap()))
        .stderr(Stdio::from(output));
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ] {
        command.env(key, &server.url);
    }
    let mut child = command.spawn().unwrap();
    let deadline = Instant::now() + Duration::from_secs(120);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!(
                "remote fixture timed out: {}",
                fs::read_to_string(&output_path).unwrap()
            );
        }
        thread::sleep(Duration::from_millis(10));
    };
    let output = fs::read_to_string(&output_path).unwrap();
    assert!(
        status.success() && output.contains("1 passed"),
        "isolated test failed or did not run: {output}"
    );
    let requests = server.requests.lock().unwrap();
    assert!(
        requests.iter().all(|(method, target)| {
            if method == "CONNECT" {
                target == "huggingface.co:443"
            } else {
                matches!(method.as_str(), "GET" | "HEAD") && target.starts_with('/')
            }
        }),
        "unexpected fixture routing: {requests:?}"
    );
    check_requests(&requests, &server.ranges.lock().unwrap());
}

pub(super) fn count(requests: &[(String, String)], method: &str, target: &str) -> usize {
    requests
        .iter()
        .filter(|(m, t)| m == method && t == target)
        .count()
}
