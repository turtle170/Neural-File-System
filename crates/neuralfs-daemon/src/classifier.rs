use std::collections::HashMap;

use anyhow::Result;
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

use crate::store::FileEntry;

const MAX_VOCAB: usize = 400;
const MAX_CLASSES: usize = 40;
const MAX_SAMPLES: usize = 6000;
const EPOCHS: usize = 150;
const LEARNING_RATE: f64 = 0.5;
const L2_REG: f64 = 0.001;
/// Per-hour decay used to *weight* which samples a retrain keeps when the index
/// is larger than `MAX_SAMPLES`. ~0.004/h is roughly a one-week half-life, so a
/// file opened last week counts about half as much as one opened today — recent
/// habits are favoured (tracking drift) without dropping long-tail history.
const RESERVOIR_LAMBDA: f64 = 0.004;
/// Weight factor for entries that have never been opened (no recency signal):
/// they can still be sampled, but are unlikely to crowd out files in active use.
const RESERVOIR_BASELINE: f64 = 0.1;

/// TF-IDF vectorizer + multinomial (softmax) logistic regression classifier.
/// Predicts which directory ("class") a query is most likely to resolve to.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Classifier {
    vocab: HashMap<String, usize>,
    idf: Vec<f64>,
    classes: Vec<String>,
    weights: Array2<f64>,
    bias: Array1<f64>,
    /// Number of online SGD updates applied since the last full (re)train.
    #[serde(default)]
    updates: u64,
    /// Monotonic version, bumped on every full train and every online update.
    /// Lets the checkpoint loop detect an evolving model cheaply.
    #[serde(default)]
    version: u64,
}

impl Classifier {
    pub fn is_trained(&self) -> bool {
        !self.classes.is_empty()
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn online_updates(&self) -> u64 {
        self.updates
    }

    pub fn num_classes(&self) -> usize {
        self.classes.len()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Apply a single online (incremental) SGD step: nudge the model toward
    /// predicting `parent_dir` for `query`. This is what lets the AI keep
    /// learning continuously from real accesses while the daemon is alive,
    /// without a full retrain. Returns true if an update was applied.
    pub fn online_update(&mut self, query: &str, parent_dir: &str, lr: f64) -> bool {
        if !self.is_trained() {
            return false;
        }
        let Some(class_idx) = self.classes.iter().position(|c| c == parent_dir) else {
            return false;
        };
        let Some(x) = self.vectorize(query) else {
            return false;
        };

        let logits = x.dot(&self.weights) + &self.bias;
        let probs = softmax_1d(&logits);

        // d = (probs - onehot(class_idx)); gradient = outer(x, d).
        let mut d = probs;
        d[class_idx] -= 1.0;

        let grad = x
            .view()
            .insert_axis(Axis(1))
            .dot(&d.view().insert_axis(Axis(0)));
        self.weights = &self.weights - lr * grad;
        self.bias = &self.bias - lr * &d;

        self.updates += 1;
        self.version += 1;
        true
    }

    /// Tokenize a path or query string into lowercase word pieces.
    pub fn tokenize(s: &str) -> Vec<String> {
        s.split(|c: char| matches!(c, '/' | '\\' | '_' | '-' | '.' | ' '))
            .map(|t| t.to_lowercase())
            .filter(|t| t.len() >= 2)
            .collect()
    }

    /// Train a fresh classifier from the current index snapshot.
    /// Labels are the top-N most common parent directories.
    pub fn train(entries: &[FileEntry]) -> Result<Classifier> {
        if entries.is_empty() {
            return Ok(Classifier::default());
        }

        let mut parent_freq: HashMap<&str, usize> = HashMap::new();
        for e in entries {
            *parent_freq.entry(e.parent.as_str()).or_insert(0) += 1;
        }
        let mut parents: Vec<(&str, usize)> = parent_freq.into_iter().collect();
        parents.sort_by(|a, b| b.1.cmp(&a.1));
        parents.truncate(MAX_CLASSES);
        let classes: Vec<String> = parents.iter().map(|(p, _)| p.to_string()).collect();
        let class_index: HashMap<&str, usize> = classes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.as_str(), i))
            .collect();

        let mut training: Vec<&FileEntry> = entries
            .iter()
            .filter(|e| class_index.contains_key(e.parent.as_str()))
            .collect();
        if training.len() > MAX_SAMPLES {
            // Down-sample to the budget with a *usage-weighted reservoir* rather
            // than a positional stride. A stride (`every Nth entry in store
            // order`) is biased by insertion order and ignores how the files are
            // actually used, so over months the model trains on an arbitrary
            // slice of one store snapshot. Weighting each entry by recency +
            // frequency instead makes the retrain represent the whole history as
            // the user actually exercises it — the thing that lets the model get
            // *better*, not just stay bounded, with long-term use.
            let now = chrono::Utc::now().timestamp();
            let seed = (now as u64)
                ^ (training.len() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            training = weighted_reservoir(&training, now, MAX_SAMPLES, seed);
        }
        if training.is_empty() || classes.len() < 2 {
            return Ok(Classifier::default());
        }

        let docs: Vec<Vec<String>> = training
            .iter()
            .map(|e| Classifier::tokenize(&e.path))
            .collect();

        let mut df: HashMap<String, usize> = HashMap::new();
        for doc in &docs {
            let mut seen = std::collections::HashSet::new();
            for tok in doc {
                if seen.insert(tok.as_str()) {
                    *df.entry(tok.clone()).or_insert(0) += 1;
                }
            }
        }

        let mut vocab_entries: Vec<(String, usize)> =
            df.into_iter().filter(|(_, c)| *c >= 2).collect();
        vocab_entries.sort_by(|a, b| b.1.cmp(&a.1));
        vocab_entries.truncate(MAX_VOCAB);

        let vocab: HashMap<String, usize> = vocab_entries
            .iter()
            .enumerate()
            .map(|(i, (tok, _))| (tok.clone(), i))
            .collect();
        let n_docs = docs.len();
        let n_features = vocab.len();
        if n_features == 0 {
            return Ok(Classifier::default());
        }

        let idf: Vec<f64> = vocab_entries
            .iter()
            .map(|(_, df)| ((n_docs as f64) / (1.0 + *df as f64)).ln() + 1.0)
            .collect();

        let mut x = Array2::<f64>::zeros((n_docs, n_features));
        let mut y = Array1::<usize>::zeros(n_docs);
        for (row, (doc, entry)) in docs.iter().zip(training.iter()).enumerate() {
            let mut counts: HashMap<usize, f64> = HashMap::new();
            for tok in doc {
                if let Some(&idx) = vocab.get(tok) {
                    *counts.entry(idx).or_insert(0.0) += 1.0;
                }
            }
            let len = doc.len().max(1) as f64;
            for (idx, count) in counts {
                x[[row, idx]] = (count / len) * idf[idx];
            }
            y[row] = class_index[entry.parent.as_str()];
        }

        let (weights, bias) = train_softmax(&x, &y, classes.len());

        Ok(Classifier {
            vocab,
            idf,
            classes,
            weights,
            bias,
            updates: 0,
            version: 1,
        })
    }

    fn vectorize(&self, query: &str) -> Option<Array1<f64>> {
        if self.vocab.is_empty() {
            return None;
        }
        let tokens = Classifier::tokenize(query);
        if tokens.is_empty() {
            return None;
        }
        let mut counts: HashMap<usize, f64> = HashMap::new();
        for tok in &tokens {
            if let Some(&idx) = self.vocab.get(tok) {
                *counts.entry(idx).or_insert(0.0) += 1.0;
            }
        }
        if counts.is_empty() {
            return None;
        }
        let len = tokens.len().max(1) as f64;
        let mut v = Array1::<f64>::zeros(self.vocab.len());
        for (idx, count) in counts {
            v[idx] = (count / len) * self.idf[idx];
        }
        Some(v)
    }

    /// Predict up to top-3 candidate directories with softmax confidence scores,
    /// sorted descending by confidence.
    pub fn predict_top3(&self, query: &str) -> Vec<(String, f64)> {
        if !self.is_trained() {
            return Vec::new();
        }
        let Some(x) = self.vectorize(query) else {
            return Vec::new();
        };
        let logits = x.dot(&self.weights) + &self.bias;
        let probs = softmax_1d(&logits);

        let mut scored: Vec<(String, f64)> = self
            .classes
            .iter()
            .cloned()
            .zip(probs.iter().cloned())
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(3);
        scored
    }
}

/// Small dependency-free PRNG (xorshift64) — used only to draw the reservoir
/// keys, so it doesn't need to be cryptographic, just well-distributed.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// A uniform draw in the open interval (0, 1).
fn unit(state: &mut u64) -> f64 {
    let u = (xorshift64(state) as f64) / (u64::MAX as f64);
    u.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON)
}

/// Sampling weight for one entry: more recent and more frequently opened files
/// weigh more, mirroring the runtime scorer (`freq * e^(-lambda * age)`). A `+1`
/// floor on frequency keeps the weight positive (every entry is sampleable),
/// and never-opened entries fall back to [`RESERVOIR_BASELINE`].
fn sample_weight(e: &FileEntry, now: i64) -> f64 {
    let recency = if e.last_open > 0 {
        let age_hours = ((now - e.last_open).max(0) as f64) / 3600.0;
        (-RESERVOIR_LAMBDA * age_hours).exp()
    } else {
        RESERVOIR_BASELINE
    };
    (e.freq as f64 + 1.0) * recency.max(1e-9)
}

/// Efraimidis–Spirakis weighted reservoir sampling (the "A-Res" scheme): give
/// each item the key `u^(1/w)` for a uniform `u` and weight `w`, then keep the
/// `k` largest keys. The result is a sample of size `min(k, n)` drawn without
/// replacement in which an item's inclusion probability scales with its weight —
/// so hot/recent files are likely to appear while the long tail still gets a
/// fair chance, all in one O(n log n) pass.
fn weighted_reservoir<'a>(
    items: &[&'a FileEntry],
    now: i64,
    k: usize,
    seed: u64,
) -> Vec<&'a FileEntry> {
    let mut state = seed | 1; // xorshift64 must not be seeded with 0
    let mut keyed: Vec<(f64, &FileEntry)> = items
        .iter()
        .map(|&e| {
            let w = sample_weight(e, now);
            (unit(&mut state).powf(1.0 / w), e)
        })
        .collect();
    keyed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    keyed.into_iter().take(k).map(|(_, e)| e).collect()
}

fn softmax_1d(logits: &Array1<f64>) -> Array1<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps = logits.mapv(|v| (v - max).exp());
    let sum: f64 = exps.sum();
    exps.mapv(|v| v / sum.max(1e-12))
}

fn softmax_rows(logits: &Array2<f64>) -> Array2<f64> {
    let mut out = logits.clone();
    for mut row in out.axis_iter_mut(Axis(0)) {
        let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        row.mapv_inplace(|v| (v - max).exp());
        let sum: f64 = row.iter().sum();
        let sum = sum.max(1e-12);
        row.mapv_inplace(|v| v / sum);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::FileEntry;

    fn entry(path: &str, freq: u64, last_open: i64) -> FileEntry {
        FileEntry {
            path: path.to_string(),
            file_name: String::new(),
            extension: String::new(),
            parent: "p".to_string(),
            size: 0,
            modified: 0,
            depth: 0,
            freq,
            last_open,
        }
    }

    #[test]
    fn reservoir_is_bounded_and_keeps_everything_when_small() {
        let now = 1_000_000;
        let items: Vec<FileEntry> = (0..10).map(|i| entry(&format!("f{i}"), 1, now)).collect();
        let refs: Vec<&FileEntry> = items.iter().collect();
        // k larger than n: keep all, no duplicates.
        let got = weighted_reservoir(&refs, now, 50, 42);
        assert_eq!(got.len(), 10);
        // k smaller than n: capped exactly at k.
        let got = weighted_reservoir(&refs, now, 4, 42);
        assert_eq!(got.len(), 4);
    }

    #[test]
    fn reservoir_favours_recent_and_frequent() {
        // One hot file (high freq, opened now) vs many cold files (never opened).
        // Across many independent draws the hot file should be picked far more
        // often than any individual cold file — the property a positional stride
        // does not give.
        let now = 1_000_000;
        let mut items = vec![entry("hot", 1000, now)];
        for i in 0..200 {
            items.push(entry(&format!("cold{i}"), 0, 0));
        }
        let refs: Vec<&FileEntry> = items.iter().collect();

        let trials = 400;
        let mut hot_hits = 0;
        for s in 0..trials {
            let got = weighted_reservoir(&refs, now, 10, s as u64 + 1);
            if got.iter().any(|e| e.path == "hot") {
                hot_hits += 1;
            }
        }
        // The hot file occupies one of 201 slots but should land in the size-10
        // sample the large majority of the time; a uniform stride would give
        // only ~10/201 ≈ 5%.
        assert!(
            hot_hits as f64 / trials as f64 > 0.5,
            "hot file selected in only {hot_hits}/{trials} draws"
        );
    }
}

fn train_softmax(x: &Array2<f64>, y: &Array1<usize>, n_classes: usize) -> (Array2<f64>, Array1<f64>) {
    let (n_samples, n_features) = x.dim();
    let mut w = Array2::<f64>::zeros((n_features, n_classes));
    let mut b = Array1::<f64>::zeros(n_classes);

    let mut y_onehot = Array2::<f64>::zeros((n_samples, n_classes));
    for (i, &c) in y.iter().enumerate() {
        y_onehot[[i, c]] = 1.0;
    }

    let n = n_samples as f64;
    for _ in 0..EPOCHS {
        let logits = x.dot(&w) + &b;
        let probs = softmax_rows(&logits);
        let diff = &probs - &y_onehot;

        let grad_w = x.t().dot(&diff) / n + L2_REG * &w;
        let grad_b = diff.mean_axis(Axis(0)).unwrap();

        w = w - LEARNING_RATE * grad_w;
        b = b - LEARNING_RATE * grad_b;
    }

    (w, b)
}
