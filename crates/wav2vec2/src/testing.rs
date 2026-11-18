//! A tiny random `Wav2Vec2ForCTC` in Hugging Face layout, for tests that
//! need no download (this crate's and the CLI's).

use anyhow::Result;
use candle_core::{Device, Tensor};
use serde_json::json;
use std::collections::HashMap;
use std::path::Path;

pub const ARPA: &str = "\\data\\
ngram 1=6
ngram 2=3

\\1-grams:
-1.0\t<unk>\t0
0\t<s>\t-0.3
-0.8\t</s>\t0
-0.5\tab\t-0.2
-0.6\tba\t-0.2
-0.9\taab\t0

\\2-grams:
-0.2\t<s> ab
-0.3\tab ba
-0.4\tba </s>

\\end\\
";

/// Writes config.json, vocab.json, tokenizer_config.json,
/// preprocessor_config.json, model.safetensors (and lm.arpa) into `dir`.
/// `stable`: XLS-R style (pre-LN, LayerNorm convs) or base style (post-LN,
/// GroupNorm on the first conv).
pub fn write_tiny_hf(dir: &Path, stable: bool, seed: u64) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let hidden = 32;
    let conv = 16;
    let layers = 2;
    let kernels = [10usize, 3, 3, 2];
    let strides = [5usize, 2, 2, 2];
    let pos_k = 8;
    let pos_g = 4;
    let ffn = 64;
    let vocab = ["<pad>", "<unk>", "|", "a", "b"];
    let config = json!({
        "architectures": ["Wav2Vec2ForCTC"],
        "hidden_size": hidden, "num_hidden_layers": layers, "num_attention_heads": 4,
        "intermediate_size": ffn, "conv_dim": vec![conv; kernels.len()],
        "conv_kernel": kernels, "conv_stride": strides, "conv_bias": stable,
        "feat_extract_norm": if stable { "layer" } else { "group" },
        "feat_extract_activation": "gelu", "hidden_act": "gelu",
        "num_conv_pos_embeddings": pos_k, "num_conv_pos_embedding_groups": pos_g,
        "do_stable_layer_norm": stable, "layer_norm_eps": 1e-5,
        "vocab_size": vocab.len() + 2, "pad_token_id": 0, "add_adapter": false,
    });
    std::fs::write(dir.join("config.json"), config.to_string())?;
    let v: serde_json::Map<String, serde_json::Value> = vocab
        .iter()
        .enumerate()
        .map(|(i, t)| (t.to_string(), json!(i)))
        .collect();
    std::fs::write(
        dir.join("vocab.json"),
        serde_json::Value::Object(v).to_string(),
    )?;
    std::fs::write(
        dir.join("added_tokens.json"),
        json!({"<s>": vocab.len(), "</s>": vocab.len() + 1}).to_string(),
    )?;
    std::fs::write(
        dir.join("tokenizer_config.json"),
        json!({"pad_token": "<pad>", "unk_token": "<unk>", "word_delimiter_token": "|"})
            .to_string(),
    )?;
    std::fs::write(
        dir.join("preprocessor_config.json"),
        json!({"sampling_rate": 16000, "do_normalize": true}).to_string(),
    )?;
    std::fs::write(dir.join("lm.arpa"), ARPA)?;

    // deterministic pseudo-random weights
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut rnd = |shape: &[usize], scale: f32| -> Result<Tensor> {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
            })
            .collect();
        Ok(Tensor::from_vec(v, shape, &Device::Cpu)?)
    };
    let mut t: HashMap<String, Tensor> = HashMap::new();
    let mut cin = 1;
    for (i, &k) in kernels.iter().enumerate() {
        let p = format!("wav2vec2.feature_extractor.conv_layers.{i}");
        t.insert(format!("{p}.conv.weight"), rnd(&[conv, cin, k], 0.5)?);
        if stable {
            t.insert(format!("{p}.conv.bias"), rnd(&[conv], 0.1)?);
        }
        if stable || i == 0 {
            t.insert(
                format!("{p}.layer_norm.weight"),
                (rnd(&[conv], 0.1)? + 1.0)?,
            );
            t.insert(format!("{p}.layer_norm.bias"), rnd(&[conv], 0.1)?);
        }
        cin = conv;
    }
    let ln = |t: &mut HashMap<String, Tensor>,
              p: &str,
              d: usize,
              rnd: &mut dyn FnMut(&[usize], f32) -> Result<Tensor>|
     -> Result<()> {
        t.insert(format!("{p}.weight"), (rnd(&[d], 0.1)? + 1.0)?);
        t.insert(format!("{p}.bias"), rnd(&[d], 0.1)?);
        Ok(())
    };
    let lin = |t: &mut HashMap<String, Tensor>,
               p: &str,
               o: usize,
               i: usize,
               rnd: &mut dyn FnMut(&[usize], f32) -> Result<Tensor>|
     -> Result<()> {
        t.insert(
            format!("{p}.weight"),
            rnd(&[o, i], 1.0 / (i as f32).sqrt())?,
        );
        t.insert(format!("{p}.bias"), rnd(&[o], 0.1)?);
        Ok(())
    };
    ln(
        &mut t,
        "wav2vec2.feature_projection.layer_norm",
        conv,
        &mut rnd,
    )?;
    lin(
        &mut t,
        "wav2vec2.feature_projection.projection",
        hidden,
        conv,
        &mut rnd,
    )?;
    let pc = "wav2vec2.encoder.pos_conv_embed.conv";
    t.insert(
        format!("{pc}.parametrizations.weight.original0"),
        (rnd(&[1, 1, pos_k], 0.2)? + 1.0)?,
    );
    t.insert(
        format!("{pc}.parametrizations.weight.original1"),
        rnd(&[hidden, hidden / pos_g, pos_k], 0.3)?,
    );
    t.insert(format!("{pc}.bias"), rnd(&[hidden], 0.1)?);
    ln(&mut t, "wav2vec2.encoder.layer_norm", hidden, &mut rnd)?;
    for l in 0..layers {
        let p = format!("wav2vec2.encoder.layers.{l}");
        for n in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            lin(
                &mut t,
                &format!("{p}.attention.{n}"),
                hidden,
                hidden,
                &mut rnd,
            )?;
        }
        ln(&mut t, &format!("{p}.layer_norm"), hidden, &mut rnd)?;
        ln(&mut t, &format!("{p}.final_layer_norm"), hidden, &mut rnd)?;
        lin(
            &mut t,
            &format!("{p}.feed_forward.intermediate_dense"),
            ffn,
            hidden,
            &mut rnd,
        )?;
        lin(
            &mut t,
            &format!("{p}.feed_forward.output_dense"),
            hidden,
            ffn,
            &mut rnd,
        )?;
    }
    lin(&mut t, "lm_head", vocab.len() + 2, hidden, &mut rnd)?;
    // a pretraining-only tensor the converter must drop
    t.insert("wav2vec2.masked_spec_embed".into(), rnd(&[hidden], 1.0)?);
    candle_core::safetensors::save(&t, dir.join("model.safetensors"))?;
    Ok(())
}

/// A second of speech-like test audio (a few chirps), 16 kHz.
pub fn test_audio(secs: f64) -> Vec<f32> {
    let n = (secs * 16_000.0) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            let env = (t * 3.0 * std::f32::consts::PI).sin().abs();
            env * ((200.0 + 400.0 * t) * t * std::f32::consts::TAU).sin() * 0.3
        })
        .collect()
}
