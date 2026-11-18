//! OpenAI-compatible HTTP API.
//!
//! - `POST /v1/audio/transcriptions` (multipart): `file` (required: wav,
//!   flac, mp3, ogg, m4a), `response_format` (`json` default, `text`,
//!   `verbose_json`, `srt`, `vtt`), `timestamp_granularities[]` (`word`
//!   adds `words` to verbose_json), `stream` (`true`: server-sent events
//!   `transcript.text.delta` per segment, then `transcript.text.done`).
//!   `model`, `language`, `prompt` and `temperature` are accepted and
//!   ignored (one model per process).
//! - `GET /health`, `GET /v1/models`.
//!
//! One transcription runs at a time; others wait their turn.

use crate::{Opened, r3};
use anyhow::{Context, Result};
use axum::extract::{DefaultBodyLimit, Multipart, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::Semaphore;
use wav2vec2::{TimedWord, Transcript, audio};

struct AppState {
    o: Opened,
    model_id: String,
    language: String,
    turn: Semaphore,
    created: u64,
}

type Shared = Arc<AppState>;

struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let kind = if self.0.is_client_error() {
            "invalid_request_error"
        } else {
            "server_error"
        };
        let body = json!({"error": {"message": self.1, "type": kind}});
        (self.0, Json(body)).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}"))
    }
}

fn bad(msg: impl Into<String>) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, msg.into())
}

pub async fn serve(
    o: Opened,
    model_id: String,
    language: String,
    host: &str,
    port: u16,
) -> Result<()> {
    let state = Arc::new(AppState {
        o,
        model_id,
        language,
        turn: Semaphore::new(1),
        created: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
    });
    let app = router(state);
    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    tracing::info!("listening on http://{addr}");
    eprintln!("wav2vec2 server on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

fn router(state: Shared) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/audio/transcriptions", post(transcriptions))
        .layer(DefaultBodyLimit::max(1 << 30))
        .with_state(state)
}

async fn health(State(st): State<Shared>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "model": st.model_id,
        "device": st.o.device,
        "language_model": st.o.rec.scorer.is_some(),
    }))
}

async fn models(State(st): State<Shared>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": st.model_id,
            "object": "model",
            "created": st.created,
            "owned_by": "wav2vec2",
        }]
    }))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    Json,
    Text,
    VerboseJson,
    Srt,
    Vtt,
}

struct Request {
    file: Vec<u8>,
    ext: Option<String>,
    format: Format,
    words: bool,
    stream: bool,
}

async fn read_form(mut mp: Multipart) -> Result<Request, ApiError> {
    let mut file = None;
    let mut ext = None;
    let mut format = Format::Json;
    let mut words = false;
    let mut stream = false;
    while let Some(field) = mp
        .next_field()
        .await
        .map_err(|e| bad(format!("bad multipart body: {e}")))?
    {
        let name = field.name().unwrap_or_default().to_string();
        if name == "file" {
            ext = field
                .file_name()
                .and_then(|f| f.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase()));
            let b = field
                .bytes()
                .await
                .map_err(|e| bad(format!("reading file: {e}")))?;
            file = Some(b.to_vec());
            continue;
        }
        let v = field
            .text()
            .await
            .map_err(|e| bad(format!("reading field '{name}': {e}")))?;
        let v = v.trim();
        match name.as_str() {
            "response_format" => {
                format = match v {
                    "" | "json" => Format::Json,
                    "text" => Format::Text,
                    "verbose_json" => Format::VerboseJson,
                    "srt" => Format::Srt,
                    "vtt" => Format::Vtt,
                    o => {
                        return Err(bad(format!(
                            "unsupported response_format '{o}' (json, text, verbose_json, srt, vtt)"
                        )));
                    }
                }
            }
            "timestamp_granularities[]" | "timestamp_granularities" => {
                if v.split(',').any(|g| g.trim() == "word") {
                    words = true;
                }
            }
            "stream" => {
                stream = matches!(v.to_ascii_lowercase().as_str(), "true" | "1" | "yes" | "on")
            }
            _ => {}
        }
    }
    let file = file
        .filter(|f| !f.is_empty())
        .ok_or_else(|| bad("'file' field is required"))?;
    Ok(Request {
        file,
        ext,
        format,
        words,
        stream,
    })
}

async fn transcriptions(State(st): State<Shared>, mp: Multipart) -> Result<Response, ApiError> {
    let req = read_form(mp).await?;
    let _turn = st.turn.acquire().await.map_err(|e| anyhow::anyhow!(e))?;
    let t = Instant::now();
    let st2 = st.clone();
    let (file, ext) = (req.file, req.ext);
    let tr = tokio::task::spawn_blocking(move || -> Result<Transcript, ApiError> {
        let rate = st2.o.rec.cfg.sample_rate;
        let wav = audio::load(file, ext.as_deref(), rate)
            .map_err(|e| bad(format!("could not decode the audio: {e:#}")))?;
        Ok(st2.o.rec.transcribe(&wav, &st2.o.opts)?)
    })
    .await
    .map_err(|e| anyhow::anyhow!("transcription task failed: {e}"))??;
    let el = t.elapsed().as_secs_f64();
    tracing::info!(
        "{:.1}s of audio in {el:.2}s ({:.0}x realtime), {} words",
        tr.duration,
        tr.duration / el.max(1e-9),
        tr.words.len()
    );
    if req.stream {
        return Ok(sse(&tr));
    }
    Ok(render(&st, &tr, req.format, req.words))
}

struct Segment {
    start: f64,
    end: f64,
    text: String,
}

/// Words grouped into segments at pauses (>= 0.6 s) or past 12 s / 30 words.
fn segments(words: &[TimedWord]) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::new();
    let mut count = 0;
    for w in words {
        let split = match out.last() {
            None => true,
            Some(s) => w.start - s.end >= 0.6 || w.end - s.start > 12.0 || count >= 30,
        };
        if split {
            out.push(Segment {
                start: w.start,
                end: w.end,
                text: w.text.clone(),
            });
            count = 1;
        } else if let Some(s) = out.last_mut() {
            s.end = w.end;
            s.text.push(' ');
            s.text.push_str(&w.text);
            count += 1;
        }
    }
    out
}

fn usage(d: f64) -> Value {
    json!({"type": "duration", "seconds": d.max(0.0).ceil() as u64})
}

fn timestamp(s: f64, sep: char) -> String {
    let ms = (s.max(0.0) * 1000.0).round() as u64;
    format!(
        "{:02}:{:02}:{:02}{sep}{:03}",
        ms / 3_600_000,
        ms / 60_000 % 60,
        ms / 1000 % 60,
        ms % 1000
    )
}

fn render(st: &AppState, tr: &Transcript, format: Format, words: bool) -> Response {
    let text_resp =
        |body: String, ct: &'static str| ([(header::CONTENT_TYPE, ct)], body).into_response();
    match format {
        Format::Json => Json(json!({"text": tr.text, "usage": usage(tr.duration)})).into_response(),
        Format::Text => text_resp(tr.text.clone(), "text/plain; charset=utf-8"),
        Format::VerboseJson => {
            let segs: Vec<Value> = segments(&tr.words)
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    json!({
                        "id": i, "seek": 0, "start": r3(s.start), "end": r3(s.end), "text": s.text,
                        "tokens": [], "temperature": 0.0, "avg_logprob": 0.0,
                        "compression_ratio": 0.0, "no_speech_prob": 0.0,
                    })
                })
                .collect();
            let mut body = json!({
                "task": "transcribe",
                "language": st.language,
                "duration": r3(tr.duration),
                "text": tr.text,
                "segments": segs,
                "usage": usage(tr.duration),
            });
            if words {
                body["words"] = tr
                    .words
                    .iter()
                    .map(|w| json!({"word": w.text, "start": r3(w.start), "end": r3(w.end)}))
                    .collect();
            }
            Json(body).into_response()
        }
        Format::Srt => {
            let mut out = String::new();
            for (i, s) in segments(&tr.words).iter().enumerate() {
                out.push_str(&format!(
                    "{}\n{} --> {}\n{}\n\n",
                    i + 1,
                    timestamp(s.start, ','),
                    timestamp(s.end, ','),
                    s.text
                ));
            }
            text_resp(out, "application/x-subrip; charset=utf-8")
        }
        Format::Vtt => {
            let mut out = String::from("WEBVTT\n\n");
            for s in segments(&tr.words) {
                out.push_str(&format!(
                    "{} --> {}\n{}\n\n",
                    timestamp(s.start, '.'),
                    timestamp(s.end, '.'),
                    s.text
                ));
            }
            text_resp(out, "text/vtt; charset=utf-8")
        }
    }
}

/// The whole transcript as server-sent events: one delta per segment, then
/// the final text (the beam search needs all the audio before it settles).
fn sse(tr: &Transcript) -> Response {
    let mut body = String::new();
    for (i, s) in segments(&tr.words).iter().enumerate() {
        let delta = if i == 0 {
            s.text.clone()
        } else {
            format!(" {}", s.text)
        };
        body.push_str(&format!(
            "data: {}\n\n",
            json!({"type": "transcript.text.delta", "delta": delta})
        ));
    }
    body.push_str(&format!(
        "data: {}\n\n",
        json!({"type": "transcript.text.done", "text": tr.text, "usage": usage(tr.duration)})
    ));
    (
        [
            (header::CONTENT_TYPE, "text/event-stream"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        body,
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(t: &str, s: f64, e: f64) -> TimedWord {
        TimedWord {
            text: t.into(),
            start: s,
            end: e,
        }
    }

    #[test]
    fn segments_split_on_pauses() {
        let ws = [w("a", 0.0, 0.3), w("b", 0.4, 0.6), w("c", 1.5, 1.8)];
        let s = segments(&ws);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].text, "a b");
        assert_eq!(s[1].text, "c");
        assert_eq!(timestamp(3725.5, ','), "01:02:05,500");
    }
}
