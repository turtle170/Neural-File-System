use std::collections::HashSet;
use std::path::PathBuf;

use crate::classifier::Classifier;
use crate::protocol::ScoredPath;
use crate::scorer;
use crate::store::{FileEntry, Store};

const MAX_BACKTRACK_LEVELS: usize = 4;

/// Implements the lookup algorithm from the design doc:
/// predict top-3 candidate directories, search each (backtracking up to
/// parent directories on miss), and fall back to a full linear scan.
pub fn find(
    store: &Store,
    classifier: &Classifier,
    query: &str,
    lambda: f64,
    max_results: usize,
) -> Vec<ScoredPath> {
    let now = chrono::Utc::now().timestamp();
    let mut visited = HashSet::new();
    let mut hits: Vec<(FileEntry, f64)> = Vec::new();

    // Fast path: if the query is exactly a filename we have indexed, return it
    // straight from the name index — no classifier, no scan. This is the
    // "drop the AI when a trivial exact match wins" shortcut.
    let q_trim = query.trim();
    if !q_trim.is_empty() {
        if let Ok(named) = store.files_named(q_trim) {
            if !named.is_empty() {
                let mut results: Vec<ScoredPath> = named
                    .into_iter()
                    .map(|entry| ScoredPath {
                        score: scorer::score(entry.freq, entry.last_open, now, lambda, 1.0),
                        path: entry.path,
                    })
                    .collect();
                results.sort_by(|a, b| {
                    b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
                });
                results.truncate(max_results);
                return results;
            }
        }
    }

    if classifier.is_trained() {
        for (dir, confidence) in classifier.predict_top3(query) {
            let mut current = PathBuf::from(&dir);
            for _ in 0..MAX_BACKTRACK_LEVELS {
                let key = current.to_string_lossy().to_lowercase();
                if !visited.insert(key) {
                    break;
                }
                let candidates = store
                    .files_in(&current.to_string_lossy())
                    .unwrap_or_default();
                let matches = fuzzy_match(query, candidates);
                if !matches.is_empty() {
                    for m in matches {
                        hits.push((m, confidence));
                    }
                    break;
                }
                match current.parent() {
                    Some(p) => current = p.to_path_buf(),
                    None => break,
                }
            }
        }
    }

    if hits.is_empty() {
        let all = store.all_entries().unwrap_or_default();
        let matches = fuzzy_match(query, all);
        for m in matches {
            hits.push((m, 0.0));
        }
    }

    let mut results: Vec<ScoredPath> = hits
        .into_iter()
        .map(|(entry, confidence)| ScoredPath {
            path: entry.path,
            score: scorer::score(entry.freq, entry.last_open, now, lambda, confidence),
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.dedup_by(|a, b| a.path == b.path);
    results.truncate(max_results);
    results
}

fn fuzzy_match(query: &str, candidates: Vec<FileEntry>) -> Vec<FileEntry> {
    let q = query.to_lowercase();
    let q_tokens: Vec<String> = Classifier::tokenize(&q);
    candidates
        .into_iter()
        .filter(|c| {
            let name = c.file_name.to_lowercase();
            if name.contains(&q) {
                return true;
            }
            if q_tokens.is_empty() {
                return false;
            }
            let name_tokens: HashSet<String> = Classifier::tokenize(&name).into_iter().collect();
            q_tokens.iter().any(|t| name_tokens.contains(t))
        })
        .collect()
}
