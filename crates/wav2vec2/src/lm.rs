//! Back-off n-gram language model read from an ARPA file, scored as KenLM's
//! `Model.BaseScore` (log10 probabilities, Katz back-off, `<unk>` for
//! out-of-vocabulary words). Pure Rust: no KenLM library.
//!
//! KenLM keeps a minimized right state (the longest suffix that extends);
//! this model keeps the last `order - 1` words. The scores are the same:
//! a context that is not an n-gram has a back-off weight of 0.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Multiply-rotate hasher for integer keys (FxHash's mixing).
#[derive(Default, Clone, Copy)]
pub struct FxHasher(u64);

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl Hasher for FxHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut b = [0u8; 8];
            b[..chunk.len()].copy_from_slice(chunk);
            self.write_u64(u64::from_le_bytes(b));
        }
    }
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(SEED);
    }
    fn write_u128(&mut self, i: u128) {
        self.write_u64(i as u64);
        self.write_u64((i >> 64) as u64);
    }
    fn write_u32(&mut self, i: u32) {
        self.write_u64(i as u64);
    }
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }
}

pub type FxBuild = BuildHasherDefault<FxHasher>;
pub type FxHashMap<K, V> = HashMap<K, V, FxBuild>;

/// Highest supported order (21 bits per word in a 128-bit key).
pub const MAX_ORDER: usize = 6;
const ID_BITS: u32 = 21;
const MAX_WORDS: usize = (1 << ID_BITS) - 2;

pub type WordId = u32;

/// Last `order - 1` words, oldest first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct State {
    words: [WordId; MAX_ORDER - 1],
    len: u8,
}

impl State {
    pub fn words(&self) -> &[WordId] {
        &self.words[..self.len as usize]
    }
}

pub struct NgramLm {
    order: usize,
    vocab: HashMap<String, WordId>,
    /// (log10 prob, log10 back-off) by n-gram key.
    entries: FxHashMap<u128, (f32, f32)>,
    pub unk: WordId,
    pub bos: WordId,
    pub eos: WordId,
    /// Unigrams listed with a back-off weight (three tab-separated fields),
    /// as pyctcdecode's `load_unigram_set_from_arpa` reads them.
    unigram_words: Vec<String>,
}

fn key(ids: &[WordId]) -> u128 {
    let mut k = 0u128;
    for (i, &id) in ids.iter().enumerate() {
        k |= ((id as u128) + 1) << (ID_BITS * i as u32);
    }
    k
}

impl NgramLm {
    pub fn from_arpa(text: &str) -> Result<Self> {
        let mut lines = text.lines().enumerate();
        // \data\ header
        let mut counts: Vec<usize> = Vec::new();
        let mut in_data = false;
        for (_, line) in lines.by_ref() {
            let l = line.trim();
            if l == "\\data\\" {
                in_data = true;
                continue;
            }
            if !in_data {
                continue;
            }
            if let Some(rest) = l.strip_prefix("ngram ") {
                let (n, c) = rest.split_once('=').context("bad 'ngram N=count' line")?;
                let n: usize = n.trim().parse()?;
                let c: usize = c.trim().parse()?;
                if n != counts.len() + 1 {
                    bail!("ARPA header: ngram orders out of sequence");
                }
                counts.push(c);
            } else if l.is_empty() && !counts.is_empty() {
                break;
            }
        }
        let order = counts.len();
        if order == 0 {
            bail!("not an ARPA file (no \\data\\ section)");
        }
        if order > MAX_ORDER {
            bail!("ARPA order {order} > {MAX_ORDER}");
        }
        if counts[0] > MAX_WORDS {
            bail!("ARPA vocabulary too large ({} words)", counts[0]);
        }
        let total: usize = counts.iter().sum();
        let mut entries: FxHashMap<u128, (f32, f32)> =
            HashMap::with_capacity_and_hasher(total + 1, FxBuild::default());
        let mut vocab: HashMap<String, WordId> = HashMap::with_capacity(counts[0] + 1);
        let mut unigram_words = Vec::new();
        let mut n = 0usize;
        let mut ids: Vec<WordId> = Vec::with_capacity(order);
        for (lineno, line) in lines {
            let l = line.trim();
            if l.is_empty() {
                continue;
            }
            if l.starts_with('\\') {
                if l == "\\end\\" {
                    break;
                }
                let m = l
                    .strip_prefix('\\')
                    .and_then(|s| s.strip_suffix("-grams:"))
                    .and_then(|s| s.parse::<usize>().ok())
                    .with_context(|| format!("line {}: unexpected '{l}'", lineno + 1))?;
                if m != n + 1 || m > order {
                    bail!("line {}: section {m}-grams out of order", lineno + 1);
                }
                n = m;
                continue;
            }
            if n == 0 {
                bail!("line {}: n-gram before any section", lineno + 1);
            }
            let mut it = l.split_whitespace();
            let prob: f32 = it
                .next()
                .and_then(|s| s.parse().ok())
                .with_context(|| format!("line {}: bad probability", lineno + 1))?;
            ids.clear();
            for _ in 0..n {
                let w = it
                    .next()
                    .with_context(|| format!("line {}: expected {n} words", lineno + 1))?;
                let id = if n == 1 {
                    let next = vocab.len() as WordId;
                    *vocab.entry(w.to_string()).or_insert(next)
                } else {
                    *vocab.get(w).with_context(|| {
                        format!("line {}: word '{w}' has no unigram", lineno + 1)
                    })?
                };
                ids.push(id);
            }
            let bow: f32 = match it.next() {
                Some(s) => s
                    .parse()
                    .with_context(|| format!("line {}: bad back-off", lineno + 1))?,
                None => 0.0,
            };
            if n == 1 && l.split('\t').count() == 3 {
                unigram_words.push(ids_word(l));
            }
            entries.insert(key(&ids), (prob, bow));
        }
        if vocab.len() > MAX_WORDS {
            bail!("ARPA vocabulary too large");
        }
        let mut lm = Self {
            order,
            unk: 0,
            bos: 0,
            eos: 0,
            vocab,
            entries,
            unigram_words,
        };
        // KenLM gives a missing <unk> probability -100
        lm.unk = lm.ensure_word("<unk>", -100.0);
        lm.bos = lm.ensure_word("<s>", -99.0);
        lm.eos = lm.ensure_word("</s>", -99.0);
        Ok(lm)
    }

    fn ensure_word(&mut self, w: &str, prob: f32) -> WordId {
        if let Some(&id) = self.vocab.get(w) {
            return id;
        }
        let id = self.vocab.len() as WordId;
        self.vocab.insert(w.to_string(), id);
        self.entries.insert(key(&[id]), (prob, 0.0));
        id
    }

    pub fn order(&self) -> usize {
        self.order
    }

    /// Vocabulary id; `<unk>` for unknown words (as KenLM's `vocab.Index`).
    pub fn index(&self, word: &str) -> WordId {
        self.vocab.get(word).copied().unwrap_or(self.unk)
    }

    /// `word in model` (KenLM): known and not `<unk>`.
    pub fn contains(&self, word: &str) -> bool {
        self.vocab.get(word).is_some_and(|&i| i != self.unk)
    }

    pub fn unigram_words(&self) -> &[String] {
        &self.unigram_words
    }

    /// The `<s>` context (KenLM `BeginSentenceWrite`).
    pub fn begin_sentence(&self) -> State {
        let mut s = State {
            words: [0; MAX_ORDER - 1],
            len: 0,
        };
        if self.order > 1 {
            s.words[0] = self.bos;
            s.len = 1;
        }
        s
    }

    /// Empty context (KenLM `NullContextWrite`).
    pub fn null_context(&self) -> State {
        State {
            words: [0; MAX_ORDER - 1],
            len: 0,
        }
    }

    /// log10 P(word | state) and the next state (KenLM `BaseScore`).
    pub fn score(&self, state: &State, word: WordId) -> (f32, State) {
        let ctx = state.words();
        let mut buf = [0 as WordId; MAX_ORDER];
        let mut backoff = 0f32;
        let mut prob = None;
        for m in (0..=ctx.len()).rev() {
            let h = &ctx[ctx.len() - m..];
            buf[..m].copy_from_slice(h);
            buf[m] = word;
            if let Some(&(p, _)) = self.entries.get(&key(&buf[..=m])) {
                prob = Some(p);
                break;
            }
            if m > 0
                && let Some(&(_, b)) = self.entries.get(&key(h))
            {
                backoff += b;
            }
        }
        // every vocabulary word has a unigram; an id outside it scores as <unk>
        let p = prob.unwrap_or_else(|| self.entries[&key(&[self.unk])].0);
        let mut next = State {
            words: [0; MAX_ORDER - 1],
            len: 0,
        };
        let keep = self.order.saturating_sub(1);
        if keep > 0 {
            let mut all: Vec<WordId> = ctx.to_vec();
            all.push(word);
            let tail = &all[all.len().saturating_sub(keep)..];
            next.words[..tail.len()].copy_from_slice(tail);
            next.len = tail.len() as u8;
        }
        (p + backoff, next)
    }
}

fn ids_word(line: &str) -> String {
    line.split('\t').nth(1).unwrap_or("").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARPA: &str = "\\data\\
ngram 1=5
ngram 2=3
ngram 3=1

\\1-grams:
-1.0\t<unk>\t0
0\t<s>\t-0.5
-0.7\t</s>\t0
-0.6\ta\t-0.3
-0.8\tb\t-0.2

\\2-grams:
-0.2\t<s> a\t-0.1
-0.4\ta b\t-0.25
-0.3\tb </s>

\\3-grams:
-0.05\t<s> a b

\\end\\
";

    fn close(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6, "{a} != {b}");
    }

    #[test]
    fn backoff_scores() {
        let lm = NgramLm::from_arpa(ARPA).unwrap();
        assert_eq!(lm.order(), 3);
        let s0 = lm.begin_sentence();
        let a = lm.index("a");
        let b = lm.index("b");
        // P(a | <s>) is a bigram
        let (p, s1) = lm.score(&s0, a);
        close(p, -0.2);
        // P(b | <s> a) is a trigram
        let (p, s2) = lm.score(&s1, b);
        close(p, -0.05);
        // P(</s> | a b): no trigram, bow(a b) + P(</s> | b)
        let (p, _) = lm.score(&s2, lm.eos);
        close(p, -0.25 + -0.3);
        // P(a | a b): bow(a b) + [no "b a"] bow(b) + P(a)
        let (p, _) = lm.score(&s2, a);
        close(p, -0.25 - 0.2 - 0.6);
        // unknown word
        let unk = lm.index("zzz");
        assert_eq!(unk, lm.unk);
        assert!(!lm.contains("zzz") && lm.contains("a") && !lm.contains("<unk>"));
        let (p, _) = lm.score(&s0, unk);
        close(p, -0.5 + -1.0);
        // unigrams with a back-off field
        let mut u = lm.unigram_words().to_vec();
        u.sort();
        assert_eq!(u, ["</s>", "<s>", "<unk>", "a", "b"]);
    }
}
