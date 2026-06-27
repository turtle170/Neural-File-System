use std::hash::Hash;
use std::sync::Arc;

use hashlink::LinkedHashMap;

/// A strict byte-bounded, frequency-aware RAM cache.
///
/// This is NeuralFS's answer to the ZFS ARC: a hot value seen more than once is
/// promoted from the *probation* segment into the *protected* segment, so it
/// survives eviction pressure that churns through once-touched data. The total
/// resident size never exceeds `cap` bytes — the cap is enforced strictly after
/// every insert and promotion.
///
/// Eviction order: probation (recency) is evicted before protected (frequency).
/// The protected segment is itself capped (default 80% of total) so that a
/// stream of new data can always be admitted without starving.
pub struct RamCache<K: Eq + Hash + Clone> {
    probation: LinkedHashMap<K, Arc<Vec<u8>>>,
    protected: LinkedHashMap<K, Arc<Vec<u8>>>,
    bytes_prob: u64,
    bytes_prot: u64,
    cap: u64,
    protected_cap: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
    promotions: u64,
}

#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub cap_bytes: u64,
    pub resident_bytes: u64,
    pub entries: usize,
    pub protected_entries: usize,
    pub probation_entries: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub promotions: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

impl<K: Eq + Hash + Clone> RamCache<K> {
    pub fn new(cap_bytes: u64) -> Self {
        Self {
            probation: LinkedHashMap::new(),
            protected: LinkedHashMap::new(),
            bytes_prob: 0,
            bytes_prot: 0,
            cap: cap_bytes.max(1),
            protected_cap: (cap_bytes / 10 * 8).max(1),
            hits: 0,
            misses: 0,
            evictions: 0,
            promotions: 0,
        }
    }

    /// Look up a value. A hit in probation promotes the entry to protected
    /// (this is the "read frequency is high enough → keep it hot" rule); a hit
    /// in protected refreshes its recency.
    pub fn get(&mut self, k: &K) -> Option<Arc<Vec<u8>>> {
        if let Some(v) = self.protected.remove(k) {
            let ret = v.clone();
            self.protected.insert(k.clone(), v); // re-insert at MRU
            self.hits += 1;
            return Some(ret);
        }
        if let Some(v) = self.probation.remove(k) {
            let len = v.len() as u64;
            self.bytes_prob -= len;
            let ret = v.clone();
            self.protected.insert(k.clone(), v);
            self.bytes_prot += len;
            self.hits += 1;
            self.promotions += 1;
            self.enforce_protected_cap();
            self.enforce_total_cap();
            return Some(ret);
        }
        self.misses += 1;
        None
    }

    /// Insert a value (typically after a miss). New data always lands in the
    /// probation segment; it earns protected status only by being read again.
    pub fn insert(&mut self, k: K, value: Arc<Vec<u8>>) {
        self.invalidate(&k);
        let len = value.len() as u64;
        self.probation.insert(k, value);
        self.bytes_prob += len;
        self.enforce_total_cap();
    }

    pub fn invalidate(&mut self, k: &K) {
        if let Some(v) = self.probation.remove(k) {
            self.bytes_prob -= v.len() as u64;
        }
        if let Some(v) = self.protected.remove(k) {
            self.bytes_prot -= v.len() as u64;
        }
    }

    pub fn contains(&self, k: &K) -> bool {
        self.protected.contains_key(k) || self.probation.contains_key(k)
    }

    pub fn clear(&mut self) {
        self.probation.clear();
        self.protected.clear();
        self.bytes_prob = 0;
        self.bytes_prot = 0;
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            cap_bytes: self.cap,
            resident_bytes: self.bytes_prob + self.bytes_prot,
            entries: self.probation.len() + self.protected.len(),
            protected_entries: self.protected.len(),
            probation_entries: self.probation.len(),
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            promotions: self.promotions,
        }
    }

    /// Demote protected LRUs back to probation until the protected segment fits.
    fn enforce_protected_cap(&mut self) {
        while self.bytes_prot > self.protected_cap {
            match self.protected.pop_front() {
                Some((k, v)) => {
                    let len = v.len() as u64;
                    self.bytes_prot -= len;
                    self.probation.insert(k, v);
                    self.bytes_prob += len;
                }
                None => break,
            }
        }
    }

    /// Strictly enforce the global byte cap: evict probation first, then protected.
    fn enforce_total_cap(&mut self) {
        while self.bytes_prob + self.bytes_prot > self.cap {
            if let Some((_, v)) = self.probation.pop_front() {
                self.bytes_prob -= v.len() as u64;
                self.evictions += 1;
            } else if let Some((_, v)) = self.protected.pop_front() {
                self.bytes_prot -= v.len() as u64;
                self.evictions += 1;
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(n: usize) -> Arc<Vec<u8>> {
        Arc::new(vec![0u8; n])
    }

    #[test]
    fn strict_cap_is_never_exceeded() {
        let mut c: RamCache<u64> = RamCache::new(10 * 1024 * 1024); // 10 MiB
        for i in 0..100u64 {
            c.insert(i, block(1024 * 1024)); // 100 x 1 MiB
            assert!(
                c.stats().resident_bytes <= 10 * 1024 * 1024,
                "cap exceeded at {i}: {}",
                c.stats().resident_bytes
            );
        }
        assert!(c.stats().entries <= 10);
    }

    #[test]
    fn frequently_read_value_survives_eviction() {
        let mut c: RamCache<String> = RamCache::new(5 * 1024 * 1024);
        c.insert("hot".into(), block(1024 * 1024));
        // Read it again -> promotes to protected.
        assert!(c.get(&"hot".into()).is_some());

        // Now stream a lot of cold, once-touched data through probation.
        for i in 0..30u64 {
            c.insert(format!("cold{i}"), block(1024 * 1024));
        }

        // The hot block, promoted by frequency, is still resident...
        assert!(c.get(&"hot".into()).is_some(), "hot value was evicted");
        // ...while an early cold block has been pushed out.
        assert!(!c.contains(&"cold0".to_string()));
    }

    #[test]
    fn hit_and_miss_accounting() {
        let mut c: RamCache<u64> = RamCache::new(1024 * 1024);
        assert!(c.get(&1).is_none()); // miss
        c.insert(1, block(1024));
        assert!(c.get(&1).is_some()); // hit
        let s = c.stats();
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 1);
        assert!((s.hit_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn value_larger_than_cap_does_not_stick() {
        let mut c: RamCache<u64> = RamCache::new(1024 * 1024);
        c.insert(1, block(4 * 1024 * 1024)); // 4 MiB into a 1 MiB cache
        assert!(c.stats().resident_bytes <= 1024 * 1024);
        assert!(!c.contains(&1));
    }
}
