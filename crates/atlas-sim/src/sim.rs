//! Replays a trace against a byte-capacity cache of weight bundles.
//!
//! Requests arrive in execution order: token by token, layer by layer. Only
//! cacheable matrices touch the cache; the rest are resident. Each policy's
//! state is private to one `simulate` call, so `simulate` is a pure function
//! of (trace, config, policy).

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt;

use serde::Serialize;

use crate::trace::Trace;

#[derive(Debug, Clone, PartialEq)]
pub enum PolicySpec {
    Lru,
    Lfu,
    /// Fixed resident set (from a calibration profile); misses bypass the cache.
    Static { name: String, hot: Vec<bool> },
    /// Farthest-next-use eviction: the offline optimum for demand caching,
    /// used as the upper bound for every online policy.
    Belady,
}

impl PolicySpec {
    pub fn name(&self) -> &str {
        match self {
            Self::Lru => "lru",
            Self::Lfu => "lfu",
            Self::Static { name, .. } => name,
            Self::Belady => "belady",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SimConfig {
    pub capacity_bytes: u64,
    /// Per matrix, in trace order.
    pub cacheable: Vec<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimError {
    CacheableMismatch { matrices: usize, flags: usize },
    HotSetMismatch { items: usize, hot: usize },
}

impl fmt::Display for SimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CacheableMismatch { matrices, flags } => {
                write!(f, "trace has {matrices} matrices but {flags} cacheable flags")
            }
            Self::HotSetMismatch { items, hot } => {
                write!(f, "trace has {items} items but the hot set has {hot}")
            }
        }
    }
}

impl std::error::Error for SimError {}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MatrixStats {
    pub name: String,
    pub cacheable: bool,
    pub bytes_per_channel: u32,
    /// Active channels / all channels: the effective density under this profile.
    pub active_fraction: f64,
    pub requested_bytes: u64,
    pub hit_bytes: u64,
    pub misses: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SimStats {
    pub policy: String,
    pub tokens: usize,
    pub capacity_bytes: u64,
    pub requested_bytes: u64,
    pub hit_bytes: u64,
    pub byte_hit_rate: f64,
    pub miss_bytes_per_token: f64,
    pub matrices: Vec<MatrixStats>,
}

trait Cache {
    /// Records an access at `tick`; returns whether it hit.
    fn access(&mut self, id: u32, tick: u64) -> bool;
}

pub fn simulate(trace: &Trace, cfg: &SimConfig, policy: &PolicySpec) -> Result<SimStats, SimError> {
    let lay = &trace.layout;
    if cfg.cacheable.len() != lay.matrices() {
        return Err(SimError::CacheableMismatch { matrices: lay.matrices(), flags: cfg.cacheable.len() });
    }
    let sizes = lay.item_sizes();
    let cap = cfg.capacity_bytes;
    Ok(match policy {
        PolicySpec::Lru => replay(trace, cfg, policy.name(), Lru::new(sizes, cap)),
        PolicySpec::Lfu => replay(trace, cfg, policy.name(), Lfu::new(sizes, cap)),
        PolicySpec::Belady => replay(trace, cfg, policy.name(), Belady::new(trace, sizes, cap)),
        PolicySpec::Static { name, hot } => {
            if hot.len() != lay.item_count() {
                return Err(SimError::HotSetMismatch { items: lay.item_count(), hot: hot.len() });
            }
            replay(trace, cfg, name, Static { hot })
        }
    })
}

fn replay(trace: &Trace, cfg: &SimConfig, name: &str, mut cache: impl Cache) -> SimStats {
    let lay = &trace.layout;
    let n = lay.matrices();
    let (mut active, mut requested, mut hits, mut misses) = (vec![0u64; n], vec![0u64; n], vec![0u64; n], vec![0u64; n]);
    for t in 0..lay.tokens {
        for l in 0..lay.layers {
            let tick = (t * lay.layers + l) as u64;
            for m in 0..n {
                let bytes = lay.bytes[m] as u64;
                for c in trace.active_channels(t, l, m) {
                    active[m] += 1;
                    if cfg.cacheable[m] {
                        requested[m] += bytes;
                        if cache.access(lay.item(l, m, c), tick) {
                            hits[m] += bytes;
                        } else {
                            misses[m] += 1;
                        }
                    }
                }
            }
        }
    }
    let requested_bytes: u64 = requested.iter().sum();
    let hit_bytes: u64 = hits.iter().sum();
    let slots = (lay.tokens * lay.layers).max(1) as f64;
    SimStats {
        policy: name.to_owned(),
        tokens: lay.tokens,
        capacity_bytes: cfg.capacity_bytes,
        requested_bytes,
        hit_bytes,
        byte_hit_rate: if requested_bytes == 0 { 1.0 } else { hit_bytes as f64 / requested_bytes as f64 },
        miss_bytes_per_token: (requested_bytes - hit_bytes) as f64 / lay.tokens.max(1) as f64,
        matrices: (0..n)
            .map(|m| MatrixStats {
                name: trace.header.matrices[m].name.clone(),
                cacheable: cfg.cacheable[m],
                bytes_per_channel: lay.bytes[m],
                active_fraction: active[m] as f64 / (slots * lay.channels[m] as f64),
                requested_bytes: requested[m],
                hit_bytes: hits[m],
                misses: misses[m],
            })
            .collect(),
    }
}

struct Static<'a> {
    hot: &'a [bool],
}

impl Cache for Static<'_> {
    fn access(&mut self, id: u32, _: u64) -> bool {
        self.hot[id as usize]
    }
}

/// Exact LRU: intrusive doubly linked list over dense item ids.
struct Lru {
    sizes: Vec<u32>,
    cap: u64,
    used: u64,
    cached: Vec<bool>,
    prev: Vec<u32>,
    next: Vec<u32>,
}

impl Lru {
    fn new(sizes: Vec<u32>, cap: u64) -> Self {
        let n = sizes.len();
        let head = n as u32; // sentinel: next = MRU, prev = LRU
        let mut prev = vec![0; n + 1];
        let mut next = vec![0; n + 1];
        prev[n] = head;
        next[n] = head;
        Self { sizes, cap, used: 0, cached: vec![false; n], prev, next }
    }

    fn head(&self) -> u32 {
        self.sizes.len() as u32
    }

    fn unlink(&mut self, id: u32) {
        let (p, n) = (self.prev[id as usize], self.next[id as usize]);
        self.next[p as usize] = n;
        self.prev[n as usize] = p;
    }

    fn push_front(&mut self, id: u32) {
        let h = self.head();
        let first = self.next[h as usize];
        self.prev[id as usize] = h;
        self.next[id as usize] = first;
        self.prev[first as usize] = id;
        self.next[h as usize] = id;
    }
}

impl Cache for Lru {
    fn access(&mut self, id: u32, _: u64) -> bool {
        if self.cached[id as usize] {
            self.unlink(id);
            self.push_front(id);
            return true;
        }
        let size = self.sizes[id as usize] as u64;
        if size > self.cap {
            return false;
        }
        self.push_front(id);
        self.cached[id as usize] = true;
        self.used += size;
        while self.used > self.cap {
            let victim = self.prev[self.head() as usize];
            self.unlink(victim);
            self.cached[victim as usize] = false;
            self.used -= self.sizes[victim as usize] as u64;
        }
        false
    }
}

/// Perfect LFU (counts survive eviction), ties broken by recency. The heap
/// holds stale entries, skipped on pop and periodically compacted.
struct Lfu {
    sizes: Vec<u32>,
    cap: u64,
    used: u64,
    cached: Vec<bool>,
    count: usize,
    freq: Vec<u32>,
    last: Vec<u64>,
    heap: BinaryHeap<Reverse<(u32, u64, u32)>>,
}

impl Lfu {
    fn new(sizes: Vec<u32>, cap: u64) -> Self {
        let n = sizes.len();
        Self { sizes, cap, used: 0, cached: vec![false; n], count: 0, freq: vec![0; n], last: vec![0; n], heap: BinaryHeap::new() }
    }

    fn entry(&self, id: u32) -> Reverse<(u32, u64, u32)> {
        Reverse((self.freq[id as usize], self.last[id as usize], id))
    }

    fn compact(&mut self) {
        if self.heap.len() > 4 * self.count + 1024 {
            self.heap = (0..self.cached.len() as u32)
                .filter(|&id| self.cached[id as usize])
                .map(|id| self.entry(id))
                .collect();
        }
    }
}

impl Cache for Lfu {
    fn access(&mut self, id: u32, tick: u64) -> bool {
        let i = id as usize;
        self.freq[i] += 1;
        self.last[i] = tick;
        let hit = self.cached[i];
        let size = self.sizes[i] as u64;
        if !hit && size > self.cap {
            return false;
        }
        if !hit {
            self.cached[i] = true;
            self.count += 1;
            self.used += size;
        }
        self.heap.push(self.entry(id));
        while self.used > self.cap {
            let Some(Reverse((f, l, v))) = self.heap.pop() else { break };
            let vi = v as usize;
            if self.cached[vi] && self.freq[vi] == f && self.last[vi] == l {
                self.cached[vi] = false;
                self.count -= 1;
                self.used -= self.sizes[vi] as u64;
            }
        }
        self.compact();
        hit
    }
}

/// Belady's MIN with bypass: evicts the cached item whose next use is
/// farthest, which may be the item just fetched. Next uses are found by
/// scanning the trace forward (expected scan ≈ 1 / firing rate).
struct Belady<'a> {
    trace: &'a Trace,
    sizes: Vec<u32>,
    cap: u64,
    used: u64,
    cached: Vec<bool>,
    count: usize,
    next_use: Vec<u64>,
    heap: BinaryHeap<(u64, u32)>,
}

impl<'a> Belady<'a> {
    fn new(trace: &'a Trace, sizes: Vec<u32>, cap: u64) -> Self {
        let n = sizes.len();
        Self { trace, sizes, cap, used: 0, cached: vec![false; n], count: 0, next_use: vec![u64::MAX; n], heap: BinaryHeap::new() }
    }

    fn next_use_after(&self, id: u32, tick: u64) -> u64 {
        let lay = &self.trace.layout;
        let (l, m, c) = lay.locate(id);
        let t0 = tick as usize / lay.layers;
        (t0 + 1..lay.tokens)
            .find(|&t| self.trace.is_active(t, l, m, c))
            .map_or(u64::MAX, |t| (t * lay.layers + l) as u64)
    }

    fn compact(&mut self) {
        if self.heap.len() > 4 * self.count + 1024 {
            self.heap = (0..self.cached.len() as u32)
                .filter(|&id| self.cached[id as usize])
                .map(|id| (self.next_use[id as usize], id))
                .collect();
        }
    }
}

impl Cache for Belady<'_> {
    fn access(&mut self, id: u32, tick: u64) -> bool {
        let i = id as usize;
        let hit = self.cached[i];
        let size = self.sizes[i] as u64;
        if !hit && size > self.cap {
            return false;
        }
        if !hit {
            self.cached[i] = true;
            self.count += 1;
            self.used += size;
        }
        self.next_use[i] = self.next_use_after(id, tick);
        self.heap.push((self.next_use[i], id));
        while self.used > self.cap {
            let Some((nu, v)) = self.heap.pop() else { break };
            let vi = v as usize;
            if self.cached[vi] && self.next_use[vi] == nu {
                self.cached[vi] = false;
                self.count -= 1;
                self.used -= self.sizes[vi] as u64;
            }
        }
        self.compact();
        hit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::tests::{header, trace_from};

    fn cfg(cap: u64) -> SimConfig {
        SimConfig { capacity_bytes: cap, cacheable: vec![true] }
    }

    /// One layer, channels of 100 B; channel sequence per token given.
    fn seq(chs: &[usize]) -> Trace {
        let chs = chs.to_vec();
        trace_from(header(chs.len() as u64, 1, &[4]), move |t, _, _, c| chs[t] == c)
    }

    fn hits(t: &Trace, cap: u64, p: &PolicySpec) -> u64 {
        simulate(t, &cfg(cap), p).unwrap().hit_bytes / 100
    }

    #[test]
    fn lru_classic_sequence() {
        // cap 2 items: a b a c b -> hits: a(3rd). c evicts b, so b misses.
        assert_eq!(hits(&seq(&[0, 1, 0, 2, 1]), 200, &PolicySpec::Lru), 1);
    }

    #[test]
    fn lfu_keeps_frequent_item() {
        // a a b c a, room for 2: c evicts b (freq 1, older), a stays (freq 2).
        assert_eq!(hits(&seq(&[0, 0, 1, 2, 0]), 200, &PolicySpec::Lfu), 2);
    }

    #[test]
    fn belady_beats_lru_on_looping_pattern() {
        // Cyclic scan over 3 items with room for 2: LRU never hits.
        let t = seq(&[0, 1, 2, 0, 1, 2, 0, 1, 2]);
        assert_eq!(hits(&t, 200, &PolicySpec::Lru), 0);
        assert!(hits(&t, 200, &PolicySpec::Belady) >= 3);
    }

    #[test]
    fn belady_is_an_upper_bound_on_random_traces() {
        let t = trace_from(header(300, 2, &[40]), |t, l, _, c| crate::rng::uniform(9, t as u64, (l * 40 + c) as u64) < 0.3 + 0.01 * c as f64);
        let cap = 100 * 30;
        let best = simulate(&t, &cfg(cap), &PolicySpec::Belady).unwrap().hit_bytes;
        for p in [PolicySpec::Lru, PolicySpec::Lfu] {
            assert!(simulate(&t, &cfg(cap), &p).unwrap().hit_bytes <= best, "{}", p.name());
        }
    }

    #[test]
    fn static_hits_only_hot_items_and_non_cacheable_are_ignored() {
        let t = trace_from(header(2, 1, &[2, 2]), |_, _, _, _| true);
        let cfg = SimConfig { capacity_bytes: 0, cacheable: vec![true, false] };
        let hot = vec![true, false, true, true];
        let s = simulate(&t, &cfg, &PolicySpec::Static { name: "s".into(), hot }).unwrap();
        assert_eq!((s.requested_bytes, s.hit_bytes), (400, 200));
        assert_eq!(s.matrices[1].requested_bytes, 0);
        assert_eq!(s.matrices[1].active_fraction, 1.0);
    }

    #[test]
    fn rejects_mismatched_config() {
        let t = seq(&[0]);
        let bad = SimConfig { capacity_bytes: 1, cacheable: vec![] };
        assert!(simulate(&t, &bad, &PolicySpec::Lru).is_err());
        let hot = PolicySpec::Static { name: "s".into(), hot: vec![true] };
        assert!(simulate(&t, &cfg(1), &hot).is_err());
    }
}
