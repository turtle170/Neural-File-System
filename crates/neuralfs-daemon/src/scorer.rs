/// score(file) = base_freq * e^(-lambda * hours_since_last_open) + classifier_confidence
///
/// Computed at query time from stored freq + last_open timestamp; never persisted.
pub fn score(freq: u64, last_open: i64, now: i64, lambda: f64, classifier_confidence: f64) -> f64 {
    if last_open <= 0 {
        return classifier_confidence;
    }
    let hours_since = ((now - last_open).max(0) as f64) / 3600.0;
    (freq as f64) * (-lambda * hours_since).exp() + classifier_confidence
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn never_opened_returns_only_classifier_confidence() {
        assert_eq!(score(0, 0, 1000, 0.1, 0.5), 0.5);
    }

    #[test]
    fn recent_open_dominates_score() {
        let now = 10_000;
        let recent = score(5, now - 3600, now, 0.1, 0.0);
        let old = score(5, now - 3600 * 100, now, 0.1, 0.0);
        assert!(recent > old);
    }

    #[test]
    fn decay_reduces_score_over_time() {
        let now = 100_000;
        let s1 = score(10, now - 3600, now, 0.1, 0.0);
        let s2 = score(10, now - 7200, now, 0.1, 0.0);
        assert!(s1 > s2);
    }
}
