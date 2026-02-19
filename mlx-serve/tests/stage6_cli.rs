#![allow(clippy::panic, clippy::unwrap_used)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

const TEST_MODEL_REPO: &str = "mlx-community/Llama-3.2-1B-Instruct-4bit";

static TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn test_lock() -> MutexGuard<'static, ()> {
    TEST_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("test lock poisoned")
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("missing workspace root")
        .to_path_buf()
}

fn release_binary_path() -> PathBuf {
    workspace_root().join("target/release/mlx-serve")
}

fn run_output(cmd: &mut Command) -> Output {
    cmd.output().expect("failed to execute command")
}

fn assert_success(output: &Output, context: &str) {
    assert!(
        output.status.success(),
        "{context} failed\nstatus: {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn wait_for_health(base_url: &str, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        let output = run_output(Command::new("curl").args(["-sS", &format!("{base_url}/health")]));
        if output.status.success() {
            return;
        }
        std::thread::sleep(Duration::from_millis(750));
    }
    panic!("timed out waiting for server health endpoint");
}

fn wait_for_process_exit(child: &mut std::process::Child, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Some(status) = child.try_wait().expect("failed to poll child process") {
            assert!(
                status.success(),
                "server exited with non-zero status: {status}"
            );
            return;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    panic!("timed out waiting for server shutdown");
}

#[test]
fn check_stage_6_cli_distribution() {
    let _guard = test_lock();
    let root = workspace_root();

    // Check 6.1 — release build + help output
    let build_release = run_output(
        Command::new("cargo")
            .arg("build")
            .arg("--release")
            .arg("-p")
            .arg("mlx-serve")
            .current_dir(&root),
    );
    assert_success(&build_release, "cargo build --release -p mlx-serve");

    let binary = release_binary_path();
    assert!(
        binary.exists(),
        "release binary not found at {}",
        binary.display()
    );

    let help_output = run_output(Command::new(&binary).arg("--help"));
    assert_success(&help_output, "mlx-serve --help");
    let help_text = String::from_utf8_lossy(&help_output.stdout);
    assert!(
        help_text.contains("serve"),
        "--help missing serve subcommand"
    );
    assert!(
        help_text.contains("generate"),
        "--help missing generate subcommand"
    );
    assert!(help_text.contains("info"), "--help missing info subcommand");

    // Check 6.2 — one-shot generate
    let generate_output = run_output(
        Command::new(&binary)
            .arg("generate")
            .arg("--model")
            .arg(TEST_MODEL_REPO)
            .arg("--prompt")
            .arg("Hello")
            .arg("--max-tokens")
            .arg("5"),
    );
    assert_success(&generate_output, "mlx-serve generate");
    let generated = String::from_utf8_lossy(&generate_output.stdout)
        .trim()
        .to_owned();
    assert!(
        !generated.is_empty(),
        "generate command produced empty stdout"
    );

    // Check 6.3 — model info
    let info_output = run_output(
        Command::new(&binary)
            .arg("info")
            .arg("--model")
            .arg(TEST_MODEL_REPO),
    );
    assert_success(&info_output, "mlx-serve info");
    let info_text = String::from_utf8_lossy(&info_output.stdout);
    assert!(
        info_text.contains("Architecture:"),
        "info output missing architecture"
    );
    assert!(
        info_text.contains("Parameter count:"),
        "info output missing parameter count"
    );
    assert!(
        info_text.contains("Estimated runtime memory"),
        "info output missing memory estimate"
    );

    // Check 6.4 — end-to-end smoke on port 9999
    let mut server = Command::new(&binary)
        .arg("serve")
        .arg("--model")
        .arg(TEST_MODEL_REPO)
        .arg("--port")
        .arg("9999")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn mlx-serve serve");

    let base_url = "http://127.0.0.1:9999";
    wait_for_health(base_url, Duration::from_secs(240));

    let payload = json!({
        "model": TEST_MODEL_REPO,
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 10,
        "stream": false
    })
    .to_string();

    let chat_output = run_output(
        Command::new("curl")
            .arg("-sS")
            .arg("-X")
            .arg("POST")
            .arg(format!("{base_url}/v1/chat/completions"))
            .arg("-H")
            .arg("Content-Type: application/json")
            .arg("-d")
            .arg(payload),
    );
    assert_success(&chat_output, "curl /v1/chat/completions");

    let chat_json: Value =
        serde_json::from_slice(&chat_output.stdout).expect("invalid JSON from chat endpoint");
    assert!(
        chat_json["choices"][0]["message"]["content"]
            .as_str()
            .is_some_and(|s| !s.is_empty()),
        "chat response content missing or empty"
    );

    let pid = server.id().to_string();
    let kill_output = run_output(Command::new("kill").arg("-TERM").arg(pid));
    assert_success(&kill_output, "kill -TERM server");
    wait_for_process_exit(&mut server, Duration::from_secs(30));
}
