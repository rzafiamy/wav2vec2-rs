//! The `wav2vec2` binary: convert, transcribe, and the HTTP API, on a tiny
//! random model.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const BIN: &str = env!("CARGO_BIN_EXE_wav2vec2");

fn setup(name: &str) -> (PathBuf, PathBuf, PathBuf) {
    let d = std::env::temp_dir().join(format!("wav2vec2-cli-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    wav2vec2::testing::write_tiny_hf(&d, true, 5).unwrap();
    let model = d.join("tiny.gguf");
    let st = Command::new(BIN)
        .args(["convert"])
        .arg(&d)
        .arg("-o")
        .arg(&model)
        .args(["--lm"])
        .arg(d.join("lm.arpa"))
        .status()
        .unwrap();
    assert!(st.success());
    let wav = d.join("a.wav");
    let bytes = wav2vec2::audio::encode_wav(&wav2vec2::testing::test_audio(1.5), 16_000).unwrap();
    std::fs::write(&wav, bytes).unwrap();
    (d, model, wav)
}

struct Server(Child);
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn http(port: u16, method: &str, path: &str, ct: &str, body: &[u8]) -> (u16, String) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Type: {ct}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    s.write_all(head.as_bytes()).unwrap();
    s.write_all(body).unwrap();
    let mut out = Vec::new();
    s.read_to_end(&mut out).unwrap();
    let text = String::from_utf8_lossy(&out).to_string();
    let status = text[9..12].parse().unwrap();
    let body = text
        .split_once("\r\n\r\n")
        .map(|x| x.1.to_string())
        .unwrap_or_default();
    (status, body)
}

fn multipart(wav: Option<&Path>, fields: &[(&str, &str)]) -> (String, Vec<u8>) {
    let b = "XyZbOuNdArY";
    let mut body = Vec::new();
    for (k, v) in fields {
        body.extend(
            format!("--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n")
                .as_bytes(),
        );
    }
    if let Some(w) = wav {
        body.extend(
            format!("--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n")
                .as_bytes(),
        );
        body.extend(std::fs::read(w).unwrap());
        body.extend(b"\r\n");
    }
    body.extend(format!("--{b}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={b}"), body)
}

#[test]
fn transcribe_cli() {
    let (_d, model, wav) = setup("cli");
    let out = Command::new(BIN)
        .args(["transcribe", "--cpu", "--json", "-m"])
        .arg(&model)
        .arg(&wav)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["duration"], 1.5);
    assert!(v["text"].is_string() && v["words"].is_array());
    let out = Command::new(BIN)
        .args(["info", "-m"])
        .arg(&model)
        .output()
        .unwrap();
    let info = String::from_utf8_lossy(&out.stdout);
    assert!(
        info.contains("language model: 3-gram") || info.contains("language model: 2-gram"),
        "{info}"
    );
}

#[test]
fn http_api() {
    let (_d, model, wav) = setup("http");
    let port = free_port();
    let _srv = Server(
        Command::new(BIN)
            .args([
                "serve",
                "--cpu",
                "--port",
                &port.to_string(),
                "--model-id",
                "tiny-mg",
                "-m",
            ])
            .arg(&model)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let t = Instant::now();
    while TcpStream::connect(("127.0.0.1", port)).is_err() {
        assert!(
            t.elapsed() < Duration::from_secs(30),
            "server did not start"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    let (s, b) = http(port, "GET", "/health", "text/plain", b"");
    assert_eq!(s, 200);
    assert!(b.contains("\"language_model\":true"), "{b}");
    let (s, b) = http(port, "GET", "/v1/models", "text/plain", b"");
    assert_eq!(s, 200);
    assert!(b.contains("tiny-mg"));

    let post = |fields: &[(&str, &str)], with_file: bool| {
        let (ct, body) = multipart(with_file.then_some(wav.as_path()), fields);
        http(port, "POST", "/v1/audio/transcriptions", &ct, &body)
    };
    let (s, b) = post(&[("model", "tiny-mg")], true);
    assert_eq!(s, 200, "{b}");
    let v: serde_json::Value = serde_json::from_str(&b).unwrap();
    assert!(v["text"].is_string());
    assert_eq!(v["usage"]["seconds"], 2);

    let (s, b) = post(
        &[
            ("response_format", "verbose_json"),
            ("timestamp_granularities[]", "word"),
        ],
        true,
    );
    assert_eq!(s, 200);
    let v: serde_json::Value = serde_json::from_str(&b).unwrap();
    assert_eq!(v["language"], "mg");
    assert_eq!(v["duration"], 1.5);
    assert!(v["words"].is_array() && v["segments"].is_array());

    let (s, b) = post(&[("response_format", "vtt")], true);
    assert_eq!(s, 200);
    assert!(b.starts_with("WEBVTT"));
    let (s, b) = post(&[("stream", "true")], true);
    assert_eq!(s, 200);
    assert!(b.contains("transcript.text.done"), "{b}");

    let (s, b) = post(&[("model", "x")], false);
    assert_eq!(s, 400);
    assert!(b.contains("'file' field is required"));
    let (s, _) = post(&[("response_format", "mp3")], true);
    assert_eq!(s, 400);
}
