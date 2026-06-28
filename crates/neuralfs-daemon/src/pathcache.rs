use std::collections::HashMap;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::protocol::ScoredPath;

/// Default time-to-live for a cached query result: 5 minutes, sliding.
pub const DEFAULT_TTL: Duration = Duration::from_secs(300);
/// Default strict RAM budget for this cache: 500 MiB, separate from the
/// filesystem's 1 GiB block cache.
pub const DEFAULT_CAP_BYTES: usize = 500 * 1024 * 1024;

/// A short-lived RAM cache for query results — the paths the AI guessed or that
/// were otherwise found. Entries live for `ttl`; the timer is *refreshed* on
/// every hit (sliding expiration), so frequently-repeated lookups stay resident
/// while one-off lookups age out. Total size is capped strictly at `cap_bytes`.
pub struct PathTtlCache {
    inner: Mutex<Inner>,
    ttl: Duration,
    cap_bytes: usize,
}

struct Inner {
    map: HashMap<String, Entry>,
    total_bytes: usize,
    hits: u64,
    misses: u64,
    expired: u64,
    evictions: u64,
}

struct Entry {
    paths: Vec<ScoredPath>,
    expires_at: Instant,
    bytes: usize,
}

#[derive(Debug, Clone)]
pub struct PathCacheStats {
    pub ttl_secs: u64,
    pub cap_bytes: usize,
    pub resident_bytes: usize,
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
    pub expired: u64,
    pub evictions: u64,
}

fn entry_bytes(query: &str, paths: &[ScoredPath]) -> usize {
    let mut b = query.len() + 64;
    for p in paths {
        b += p.path.len() + 24;
    }
    b
}

impl PathTtlCache {
    pub fn new(ttl: Duration, cap_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                total_bytes: 0,
                hits: 0,
                misses: 0,
                expired: 0,
                evictions: 0,
            }),
            ttl,
            cap_bytes,
        }
    }

    /// Look up a query. On a hit the entry's expiry is pushed out by `ttl`
    /// (sliding), so being queried again within the window keeps it alive.
    pub fn get(&self, query: &str, now: Instant) -> Option<Vec<ScoredPath>> {
        let mut g = self.inner.lock();
        // 1 = alive, 2 = expired, 0 = absent.
        let state = match g.map.get(query) {
            Some(e) if e.expires_at > now => 1,
            Some(_) => 2,
            None => 0,
        };
        match state {
            1 => {
                let e = g.map.get_mut(query).unwrap();
                e.expires_at = now + self.ttl; // sliding refresh
                let paths = e.paths.clone();
                g.hits += 1;
                Some(paths)
            }
            2 => {
                let e = g.map.remove(query).unwrap();
                g.total_bytes -= e.bytes;
                g.expired += 1;
                g.misses += 1;
                None
            }
            _ => {
                g.misses += 1;
                None
            }
        }
    }

    /// Cache a non-empty result set for `query`, evicting the soonest-to-expire
    /// entries if the 500 MiB budget would be exceeded.
    pub fn insert(&self, query: String, paths: Vec<ScoredPath>, now: Instant) {
        if paths.is_empty() {
            return;
        }
        let bytes = entry_bytes(&query, &paths);
        if bytes > self.cap_bytes {
            return; // a single entry larger than the whole budget: skip.
        }
        let mut g = self.inner.lock();
        if let Some(old) = g.map.remove(&query) {
            g.total_bytes -= old.bytes;
        }
        while g.total_bytes + bytes > self.cap_bytes {
            // Evict the entry with the earliest expiry (closest to death).
            let victim = g
                .map
                .iter()
                .min_by_key(|(_, e)| e.expires_at)
                .map(|(k, _)| k.clone());
            match victim {
                Some(k) => {
                    if let Some(e) = g.map.remove(&k) {
                        g.total_bytes -= e.bytes;
                        g.evictions += 1;
                    }
                }
                None => break,
            }
        }
        g.total_bytes += bytes;
        g.map.insert(
            query,
            Entry {
                paths,
                expires_at: now + self.ttl,
                bytes,
            },
        );
    }

    /// Drop every entry whose TTL has elapsed. Called periodically so memory is
    /// reclaimed even for queries that are never repeated.
    pub fn sweep(&self, now: Instant) {
        let mut g = self.inner.lock();
        let dead: Vec<String> = g
            .map
            .iter()
            .filter(|(_, e)| e.expires_at <= now)
            .map(|(k, _)| k.clone())
            .collect();
        for k in dead {
            if let Some(e) = g.map.remove(&k) {
                g.total_bytes -= e.bytes;
                g.expired += 1;
            }
        }
    }

    pub fn stats(&self) -> PathCacheStats {
        let g = self.inner.lock();
        PathCacheStats {
            ttl_secs: self.ttl.as_secs(),
            cap_bytes: self.cap_bytes,
            resident_bytes: g.total_bytes,
            entries: g.map.len(),
            hits: g.hits,
            misses: g.misses,
            expired: g.expired,
            evictions: g.evictions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(path: &str) -> ScoredPath {
        ScoredPath {
            path: path.to_string(),
            score: 1.0,
        }
    }

    #[test]
    fn hit_then_expire() {
        let c = PathTtlCache::new(Duration::from_secs(300), DEFAULT_CAP_BYTES);
        let t0 = Instant::now();
        c.insert("report".into(), vec![p("C:/a/report.txt")], t0);

        // Within the window -> hit.
        assert!(c.get("report", t0 + Duration::from_secs(60)).is_some());
        // Past the (un-refreshed-from-t0) window -> but we DID refresh at +60s,
        // so expiry is now t0+360s. At t0+300s it is still alive.
        assert!(c.get("report", t0 + Duration::from_secs(300)).is_some());
        // Long after the last access -> expired.
        assert!(c.get("report", t0 + Duration::from_secs(700)).is_none());
    }

    #[test]
    fn timer_refreshes_on_repeat_query() {
        let c = PathTtlCache::new(Duration::from_secs(300), DEFAULT_CAP_BYTES);
        let t0 = Instant::now();
        c.insert("q".into(), vec![p("/x")], t0);

        // Keep querying every 4 minutes; each refreshes the 5-minute timer, so
        // it never dies despite total elapsed time far exceeding 5 minutes.
        let mut now = t0;
        for _ in 0..10 {
            now += Duration::from_secs(240);
            assert!(c.get("q", now).is_some(), "evaporated at {:?}", now - t0);
        }
    }

    #[test]
    fn sweep_reclaims_expired() {
        let c = PathTtlCache::new(Duration::from_secs(300), DEFAULT_CAP_BYTES);
        let t0 = Instant::now();
        c.insert("a".into(), vec![p("/a")], t0);
        c.insert("b".into(), vec![p("/b")], t0);
        assert_eq!(c.stats().entries, 2);
        c.sweep(t0 + Duration::from_secs(301));
        assert_eq!(c.stats().entries, 0);
        assert_eq!(c.stats().resident_bytes, 0);
    }

    #[test]
    fn strict_cap_enforced() {
        // ~1 KiB budget; each entry ~ path.len()+24+query+64.
        let c = PathTtlCache::new(Duration::from_secs(300), 2000);
        let t0 = Instant::now();
        for i in 0..200 {
            let big = "x".repeat(100);
            c.insert(format!("q{i}"), vec![p(&big)], t0 + Duration::from_secs(i));
            assert!(
                c.stats().resident_bytes <= 2000,
                "cap exceeded: {}",
                c.stats().resident_bytes
            );
        }
    }

    #[test]
    fn empty_results_not_cached() {
        let c = PathTtlCache::new(Duration::from_secs(300), DEFAULT_CAP_BYTES);
        let t0 = Instant::now();
        c.insert("nope".into(), vec![], t0);
        assert_eq!(c.stats().entries, 0);
    }
}
