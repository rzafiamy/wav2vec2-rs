//! CTC decoding: greedy (argmax) and pyctcdecode's beam search with an
//! n-gram language model (shallow fusion), ported to give the same results
//! as `pyctcdecode.BeamSearchDecoderCTC.decode` 0.5 on character
//! vocabularies:
//!
//! - log-softmax, clipped to log(1e-15);
//! - per frame, the characters at or above `token_min_logp` (and the
//!   argmax) extend every beam; beams with the same text, partial word and
//!   last character merge (log-sum-exp of their acoustic scores);
//! - a word is scored when a space closes it:
//!   `alpha * ln(10) * (log10 P(word | context) [+ unk_offset if OOV]) + beta`,
//!   plus `log10 P(</s> | ...)` for the last word; a partial word that no
//!   known word starts with costs `unk_offset` (scaled by length / 6 past 6
//!   characters);
//! - beams below best + `beam_prune_logp` are dropped, the best
//!   `beam_width` kept.

use crate::config::Vocab;
use crate::lm::{FxHashMap, NgramLm, State, WordId};
use std::collections::HashMap;
use std::sync::Arc;

pub const MIN_TOKEN_CLIP_P: f64 = 1e-15;
const LOG_BASE_CHANGE_FACTOR: f64 = std::f64::consts::LN_10;
const AVG_TOKEN_LEN: usize = 6;

/// A recognized word with its frame span (end exclusive).
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub text: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Decoded {
    pub text: String,
    pub words: Vec<Word>,
}

/// Row-wise log-softmax of `[T, V]` logits (f32, as numpy), clipped.
pub fn log_softmax(logits: &[f32], vocab: usize) -> Vec<f32> {
    let clip = MIN_TOKEN_CLIP_P.ln() as f32;
    let mut out = Vec::with_capacity(logits.len());
    for row in logits.chunks_exact(vocab) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let max = if max.is_finite() { max } else { 0.0 };
        let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
        let ls = sum.ln();
        out.extend(row.iter().map(|&x| (x - max - ls).clamp(clip, 0.0)));
    }
    out
}

/// Greedy CTC: argmax per frame, repeats collapsed, blanks removed; words
/// split on the delimiter.
pub fn greedy(logits: &[f32], vocab: &Vocab) -> Decoded {
    let v = vocab.labels.len();
    let mut words: Vec<Word> = Vec::new();
    let mut cur = String::new();
    let mut span = (0usize, 0usize);
    let mut prev = usize::MAX;
    for (t, row) in logits.chunks_exact(v).enumerate() {
        let mut best = 0;
        for (i, &x) in row.iter().enumerate() {
            if x > row[best] {
                best = i;
            }
        }
        if best != prev && best != vocab.blank_id {
            let label = &vocab.labels[best];
            if label == " " {
                if !cur.is_empty() {
                    words.push(Word {
                        text: std::mem::take(&mut cur),
                        start: span.0,
                        end: span.1,
                    });
                }
            } else {
                if cur.is_empty() {
                    span.0 = t;
                }
                cur.push_str(label);
                span.1 = t + 1;
            }
        } else if best == prev && best != vocab.blank_id && !cur.is_empty() {
            span.1 = t + 1;
        }
        prev = best;
    }
    if !cur.is_empty() {
        words.push(Word {
            text: cur,
            start: span.0,
            end: span.1,
        });
    }
    finish(words)
}

fn finish(words: Vec<Word>) -> Decoded {
    let words: Vec<Word> = words
        .into_iter()
        .flat_map(|w| {
            // a label may hold spaces (e.g. "<s>"-like tokens never do, but
            // be safe): keep words space-free
            let parts: Vec<String> = w.text.split_whitespace().map(String::from).collect();
            let (s, e) = (w.start, w.end);
            parts.into_iter().map(move |p| Word {
                text: p,
                start: s,
                end: e,
            })
        })
        .collect();
    let text = words
        .iter()
        .map(|w| w.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    Decoded { text, words }
}

#[derive(Debug, Clone, Copy)]
pub struct BeamParams {
    pub beam_width: usize,
    pub beam_prune_logp: f64,
    pub token_min_logp: f64,
    pub alpha: f64,
    pub beta: f64,
    pub unk_score_offset: f64,
    pub score_boundary: bool,
    /// Keep one beam per (last n-1 words, partial word, last character):
    /// what `decode()` does (`decode_beams()` does not).
    pub prune_history: bool,
}

impl Default for BeamParams {
    fn default() -> Self {
        Self {
            beam_width: 100,
            beam_prune_logp: -10.0,
            token_min_logp: -5.0,
            alpha: 0.5,
            beta: 1.5,
            unk_score_offset: -10.0,
            score_boundary: true,
            prune_history: true,
        }
    }
}

/// The language model with pyctcdecode's unigram set (ARPA unigrams that
/// the model knows), sorted for prefix lookups.
pub struct Scorer {
    pub lm: NgramLm,
    unigrams: Vec<String>,
}

impl Scorer {
    pub fn new(lm: NgramLm) -> Self {
        let mut unigrams: Vec<String> = lm
            .unigram_words()
            .iter()
            .filter(|w| lm.contains(w))
            .cloned()
            .collect();
        if unigrams.is_empty() {
            // pyctcdecode refuses such a file; fall back to "no unigram set"
            tracing::warn!("no unigrams with a back-off field in the ARPA file");
        }
        unigrams.sort();
        unigrams.dedup();
        Self { lm, unigrams }
    }

    pub fn from_arpa(text: &str) -> anyhow::Result<Self> {
        Ok(Self::new(NgramLm::from_arpa(text)?))
    }

    fn in_unigrams(&self, w: &str) -> bool {
        self.unigrams
            .binary_search_by(|u| u.as_str().cmp(w))
            .is_ok()
    }

    /// Some unigram starts with `p` (pygtrie `has_node`).
    fn has_prefix(&self, p: &str) -> bool {
        let i = self.unigrams.partition_point(|u| u.as_str() < p);
        self.unigrams.get(i).is_some_and(|u| u.starts_with(p))
    }

    fn partial_score(&self, p: &str, params: &BeamParams) -> f64 {
        let oov = if self.unigrams.is_empty() || !self.has_prefix(p) {
            1.0
        } else {
            0.0
        };
        let mut s = params.unk_score_offset * oov;
        let n = p.chars().count();
        if n > AVG_TOKEN_LEN {
            s = s * n as f64 / AVG_TOKEN_LEN as f64;
        }
        s
    }

    fn start_state(&self, params: &BeamParams) -> State {
        if params.score_boundary {
            self.lm.begin_sentence()
        } else {
            self.lm.null_context()
        }
    }

    /// `LanguageModel.score`: the fused score of `word` and the next state.
    fn word_score(
        &self,
        state: &State,
        word: &str,
        last: bool,
        params: &BeamParams,
    ) -> (f64, State) {
        let id: WordId = self.lm.index(word);
        let (p, end) = self.lm.score(state, id);
        let mut s = p as f64;
        if (!self.unigrams.is_empty() && !self.in_unigrams(word)) || !self.lm.contains(word) {
            s += params.unk_score_offset;
        }
        if last && params.score_boundary {
            s += self.lm.score(&end, self.lm.eos).0 as f64;
        }
        (params.alpha * s * LOG_BASE_CHANGE_FACTOR + params.beta, end)
    }
}

const NONE: u32 = u32::MAX;

/// Interned strings and prefix trees: beams carry small ids.
struct Arena {
    words: Vec<String>,
    word_ids: HashMap<String, u32>,
    /// text trie: (parent text, word)
    text_nodes: Vec<(u32, u32)>,
    text_children: FxHashMap<(u32, u32), u32>,
    /// partial-word trie: (parent, label), with its string
    part_nodes: Vec<(u32, u32)>,
    part_str: Vec<Arc<str>>,
    part_children: FxHashMap<(u32, u32), u32>,
    part_word: Vec<u32>,
    /// word frame spans: persistent lists (parent, start, end)
    spans: Vec<(u32, i32, i32)>,
}

impl Arena {
    fn new() -> Self {
        Self {
            words: vec![String::new()],
            word_ids: HashMap::from([(String::new(), 0)]),
            text_nodes: vec![(NONE, NONE)],
            text_children: FxHashMap::default(),
            part_nodes: vec![(NONE, NONE)],
            part_str: vec![Arc::from("")],
            part_children: FxHashMap::default(),
            part_word: vec![0],
            spans: vec![(NONE, -1, -1)],
        }
    }

    /// `_merge_tokens(text, word)`.
    fn merge(&mut self, text: u32, word: u32) -> u32 {
        if word == 0 {
            return text;
        }
        let next = self.text_nodes.len() as u32;
        *self.text_children.entry((text, word)).or_insert_with(|| {
            self.text_nodes.push((text, word));
            next
        })
    }

    fn extend_part(&mut self, part: u32, label: u32, s: &str) -> u32 {
        if let Some(&c) = self.part_children.get(&(part, label)) {
            return c;
        }
        let id = self.part_nodes.len() as u32;
        let mut string = String::with_capacity(self.part_str[part as usize].len() + s.len());
        string.push_str(&self.part_str[part as usize]);
        string.push_str(s);
        self.part_nodes.push((part, label));
        self.part_str.push(Arc::from(string));
        self.part_word.push(NONE);
        self.part_children.insert((part, label), id);
        id
    }

    /// The word a partial word becomes.
    fn part_as_word(&mut self, part: u32) -> u32 {
        let w = self.part_word[part as usize];
        if w != NONE {
            return w;
        }
        let s = self.part_str[part as usize].to_string();
        let id = match self.word_ids.get(&s) {
            Some(&i) => i,
            None => {
                let i = self.words.len() as u32;
                self.words.push(s.clone());
                self.word_ids.insert(s, i);
                i
            }
        };
        self.part_word[part as usize] = id;
        id
    }

    fn push_span(&mut self, list: u32, span: (i32, i32)) -> u32 {
        self.spans.push((list, span.0, span.1));
        (self.spans.len() - 1) as u32
    }

    fn text_words(&self, mut t: u32) -> Vec<u32> {
        let mut v = Vec::new();
        while t != 0 {
            let (p, w) = self.text_nodes[t as usize];
            v.push(w);
            t = p;
        }
        v.reverse();
        v
    }

    fn span_list(&self, mut s: u32) -> Vec<(i32, i32)> {
        let mut v = Vec::new();
        while s != 0 {
            let (p, a, b) = self.spans[s as usize];
            v.push((a, b));
            s = p;
        }
        v.reverse();
        v
    }
}

#[derive(Clone, Copy, Debug)]
struct Beam {
    text: u32,
    next_word: u32,
    part: u32,
    /// label index; NONE = None
    last: u32,
    spans: u32,
    part_span: (i32, i32),
    logit: f64,
}

#[derive(Clone, Copy)]
struct LmBeam {
    beam: Beam,
    score: f64,
}

/// pyctcdecode's beam search over `[T, V]` raw logits. Without a scorer
/// this is a plain CTC prefix beam search.
pub fn beam_search(
    logits: &[f32],
    vocab: &Vocab,
    scorer: Option<&Scorer>,
    params: &BeamParams,
) -> Decoded {
    beam_search_nbest(logits, vocab, scorer, params, 1)
        .into_iter()
        .next()
        .map(|b| b.decoded)
        .unwrap_or_default()
}

/// A final beam: the transcript, its acoustic score and its combined score.
#[derive(Debug, Clone)]
pub struct Hypothesis {
    pub decoded: Decoded,
    pub logit_score: f64,
    pub score: f64,
}

/// The `n` best final beams (pyctcdecode's `decode_beams`, best first).
pub fn beam_search_nbest(
    logits: &[f32],
    vocab: &Vocab,
    scorer: Option<&Scorer>,
    params: &BeamParams,
    n: usize,
) -> Vec<Hypothesis> {
    let v = vocab.labels.len();
    let logp = log_softmax(logits, v);
    let mut arena = Arena::new();
    let labels: Vec<&str> = vocab.labels.iter().map(String::as_str).collect();
    // LM cache: (text, eos) -> (raw score, state)
    let mut lm_cache: FxHashMap<(u32, bool), (f64, State)> = FxHashMap::default();
    if let Some(s) = scorer {
        lm_cache.insert((0, false), (0.0, s.start_state(params)));
    }
    let mut part_cache: FxHashMap<u32, f64> = FxHashMap::default();

    let mut beams = vec![Beam {
        text: 0,
        next_word: 0,
        part: 0,
        last: NONE,
        spans: 0,
        part_span: (-1, -1),
        logit: 0.0,
    }];
    let mut idx_list: Vec<usize> = Vec::with_capacity(v);
    let mut new_beams: Vec<Beam> = Vec::new();
    for (t, col) in logp.chunks_exact(v).enumerate() {
        let mut max_idx = 0;
        for (i, &x) in col.iter().enumerate() {
            if x > col[max_idx] {
                max_idx = i;
            }
        }
        idx_list.clear();
        for (i, &x) in col.iter().enumerate() {
            if x as f64 >= params.token_min_logp || i == max_idx {
                idx_list.push(i);
            }
        }
        new_beams.clear();
        let fi = t as i32;
        for &ci in &idx_list {
            let p = col[ci] as f64;
            let ch = labels[ci];
            for b in &beams {
                let nb = if ch.is_empty() || b.last == ci as u32 {
                    Beam {
                        last: ci as u32,
                        part_span: if ch.is_empty() {
                            b.part_span
                        } else {
                            (b.part_span.0, fi + 1)
                        },
                        logit: b.logit + p,
                        ..*b
                    }
                } else if ch == " " {
                    let spans = if b.part == 0 {
                        b.spans
                    } else {
                        arena.push_span(b.spans, b.part_span)
                    };
                    Beam {
                        text: b.text,
                        next_word: arena.part_as_word(b.part),
                        part: 0,
                        last: ci as u32,
                        spans,
                        part_span: (-1, -1),
                        logit: b.logit + p,
                    }
                } else {
                    Beam {
                        part: arena.extend_part(b.part, ci as u32, ch),
                        last: ci as u32,
                        part_span: if b.part_span.0 < 0 {
                            (fi, fi + 1)
                        } else {
                            (b.part_span.0, fi + 1)
                        },
                        logit: b.logit + p,
                        ..*b
                    }
                };
                new_beams.push(nb);
            }
        }
        let merged = merge_beams(&mut arena, &new_beams);
        let scored = lm_beams(
            &mut arena,
            &merged,
            scorer,
            params,
            &mut lm_cache,
            &mut part_cache,
            false,
        );
        let trimmed = prune(scored, params);
        beams = if params.prune_history {
            let order = scorer.map_or(1, |s| s.lm.order());
            prune_history(&arena, trimmed, order.saturating_sub(1).max(1))
        } else {
            trimmed.into_iter().map(|b| b.beam).collect()
        };
    }
    // close the last word
    let mut last_beams = Vec::with_capacity(beams.len());
    for b in &beams {
        let spans = if b.part == 0 {
            b.spans
        } else {
            arena.push_span(b.spans, b.part_span)
        };
        last_beams.push(Beam {
            text: b.text,
            next_word: arena.part_as_word(b.part),
            part: 0,
            last: NONE,
            spans,
            part_span: (-1, -1),
            logit: b.logit,
        });
    }
    let merged = merge_beams(&mut arena, &last_beams);
    let scored = lm_beams(
        &mut arena,
        &merged,
        scorer,
        params,
        &mut lm_cache,
        &mut part_cache,
        true,
    );
    let trimmed = prune(scored, params);
    trimmed
        .iter()
        .take(n)
        .map(|best| {
            let ids = arena.text_words(best.beam.text);
            let spans = arena.span_list(best.beam.spans);
            let words = ids
                .iter()
                .zip(spans.iter().copied().chain(std::iter::repeat((-1, -1))))
                .map(|(&w, (s, e))| Word {
                    text: arena.words[w as usize].clone(),
                    start: s.max(0) as usize,
                    end: e.max(0) as usize,
                })
                .collect();
            Hypothesis {
                decoded: finish(words),
                logit_score: best.beam.logit,
                score: best.score,
            }
        })
        .collect()
}

fn merge_beams(arena: &mut Arena, beams: &[Beam]) -> Vec<Beam> {
    let mut index: FxHashMap<(u32, u32, u32), usize> = FxHashMap::default();
    let mut out: Vec<Beam> = Vec::with_capacity(beams.len());
    for b in beams {
        let text = arena.merge(b.text, b.next_word);
        match index.get(&(text, b.part, b.last)) {
            Some(&i) => {
                // as pyctcdecode: the dict slot keeps its position but takes
                // the latest beam's fields (text / next word decide how the
                // LM scores it), with the summed score
                let logit = sum_log_scores(out[i].logit, b.logit);
                out[i] = Beam { logit, ..*b };
            }
            None => {
                index.insert((text, b.part, b.last), out.len());
                out.push(*b);
            }
        }
    }
    out
}

/// `_prune_history`: the first (best) beam per (last `n` words, partial
/// word, last character).
fn prune_history(arena: &Arena, beams: Vec<LmBeam>, n: usize) -> Vec<Beam> {
    let mut seen: std::collections::HashSet<(Vec<u32>, u32, u32), crate::lm::FxBuild> =
        Default::default();
    let mut out = Vec::with_capacity(beams.len());
    for b in beams {
        let mut tail = Vec::with_capacity(n);
        let mut t = b.beam.text;
        while t != 0 && tail.len() < n {
            let (p, w) = arena.text_nodes[t as usize];
            tail.push(w);
            t = p;
        }
        if seen.insert((tail, b.beam.part, b.beam.last)) {
            out.push(b.beam);
        }
    }
    out
}

fn sum_log_scores(s1: f64, s2: f64) -> f64 {
    if s1 >= s2 {
        s1 + (1.0 + (s2 - s1).exp()).ln()
    } else {
        s2 + (1.0 + (s1 - s2).exp()).ln()
    }
}

fn lm_beams(
    arena: &mut Arena,
    beams: &[Beam],
    scorer: Option<&Scorer>,
    params: &BeamParams,
    lm_cache: &mut FxHashMap<(u32, bool), (f64, State)>,
    part_cache: &mut FxHashMap<u32, f64>,
    eos: bool,
) -> Vec<LmBeam> {
    let mut out = Vec::with_capacity(beams.len());
    for b in beams {
        let new_text = arena.merge(b.text, b.next_word);
        let mut score = b.logit;
        if let Some(sc) = scorer {
            let lm_score = match lm_cache.get(&(new_text, eos)) {
                Some(&(s, _)) => s,
                None => {
                    let (prev, state) = lm_cache[&(b.text, false)];
                    let word = &arena.words[b.next_word as usize];
                    let (s, end) = sc.word_score(&state, word, eos, params);
                    let raw = prev + s;
                    lm_cache.insert((new_text, eos), (raw, end));
                    raw
                }
            };
            let mut lm_score = lm_score;
            if b.part != 0 {
                let ps = *part_cache
                    .entry(b.part)
                    .or_insert_with(|| sc.partial_score(&arena.part_str[b.part as usize], params));
                lm_score += ps;
            }
            score += lm_score;
        }
        out.push(LmBeam {
            beam: Beam {
                text: new_text,
                next_word: 0,
                ..*b
            },
            score,
        });
    }
    out
}

fn prune(mut beams: Vec<LmBeam>, params: &BeamParams) -> Vec<LmBeam> {
    let Some(max) = beams.iter().map(|b| b.score).reduce(f64::max) else {
        return beams;
    };
    beams.retain(|b| b.score >= max + params.beam_prune_logp);
    // heapq.nlargest: stable for ties
    beams.sort_by(|a, b| b.score.total_cmp(&a.score));
    beams.truncate(params.beam_width.max(1));
    beams
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> Vocab {
        Vocab {
            labels: vec!["".into(), " ".into(), "a".into(), "b".into()],
            blank_id: 0,
            word_delimiter_id: Some(1),
        }
    }

    /// One-hot-ish logits from a frame label sequence.
    fn logits(seq: &[usize]) -> Vec<f32> {
        let mut v = Vec::new();
        for &s in seq {
            for i in 0..4 {
                v.push(if i == s { 10.0 } else { 0.0 });
            }
        }
        v
    }

    #[test]
    fn greedy_collapses() {
        // a a _ a b _ " " b b
        let l = logits(&[2, 2, 0, 2, 3, 0, 1, 3, 3]);
        let d = greedy(&l, &vocab());
        assert_eq!(d.text, "aab b");
        assert_eq!(d.words[0].start, 0);
        assert_eq!(d.words[0].end, 5);
        assert_eq!(d.words[1].start, 7);
        assert_eq!(d.words[1].end, 9);
    }

    #[test]
    fn beam_without_lm_matches_greedy() {
        let l = logits(&[2, 2, 0, 2, 3, 0, 1, 3, 3]);
        let d = beam_search(&l, &vocab(), None, &BeamParams::default());
        assert_eq!(d.text, "aab b");
        assert_eq!((d.words[1].start, d.words[1].end), (7, 9));
    }

    #[test]
    fn lm_prefers_known_word() {
        // acoustics slightly prefer "ab", the LM only knows "ba"
        let arpa = "\\data\\\nngram 1=4\n\n\\1-grams:\n-1\t<unk>\t0\n0\t<s>\t0\n-0.5\t</s>\t0\n-0.3\tba\t0\n\n\\end\\\n";
        let sc = Scorer::from_arpa(arpa).unwrap();
        let mut l = Vec::new();
        for (x, y) in [(2, 3), (3, 2)] {
            let mut row = vec![0f32; 4];
            row[x] = 5.0;
            row[y] = 4.5;
            l.extend(row);
        }
        let p = BeamParams {
            alpha: 1.0,
            beta: 0.0,
            ..Default::default()
        };
        assert_eq!(beam_search(&l, &vocab(), None, &p).text, "ab");
        assert_eq!(beam_search(&l, &vocab(), Some(&sc), &p).text, "ba");
    }
}
