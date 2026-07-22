/// Rust guideline compliant 2026-06-17
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashSet};

/// HNSW configuration parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnConfig {
    /// Enables HNSW ANN indexing.
    pub enabled: bool,
    /// Minimum vector count threshold to use ANN instead of linear scan.
    pub threshold: usize,
    /// Maximum outgoing connections per node per level.
    pub m: usize,
    /// Search beam size during construction.
    pub ef_construction: usize,
    /// Search beam size during query time.
    pub ef_search: usize,
}

impl Default for AnnConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 500,
            m: 16,
            ef_construction: 64,
            ef_search: 32,
        }
    }
}

/// Computes cosine distance between two normalized or raw vectors.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-6);
    let sim = (dot / denom).clamp(-1.0, 1.0);
    1.0 - sim
}

#[derive(Debug, Clone, PartialEq)]
struct Candidate {
    distance: f32,
    node_id: usize,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap ordering for nearest neighbor priority queue
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, PartialEq)]
struct MinCandidate {
    distance: f32,
    node_id: usize,
}

impl Eq for MinCandidate {}

impl Ord for MinCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for MinCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// A node in the HNSW graph containing neighbor lists per level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswNode {
    pub level: usize,
    pub neighbors: Vec<Vec<usize>>,
}

/// Hierarchical Navigable Small World (HNSW) graph vector index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswIndex {
    pub entry_point: Option<usize>,
    pub max_level: usize,
    pub m: usize,
    pub m0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub nodes: Vec<HnswNode>,
}

impl HnswIndex {
    /// Builds an HNSW index from a set of vectors.
    pub fn build(vectors: &[Vec<f32>], config: &AnnConfig) -> Self {
        let m = config.m.max(2);
        let m0 = m * 2;
        let ef_construction = config.ef_construction.max(m);
        let ef_search = config.ef_search.max(1);

        let mut index = Self {
            entry_point: None,
            max_level: 0,
            m,
            m0,
            ef_construction,
            ef_search,
            nodes: Vec::with_capacity(vectors.len()),
        };

        if vectors.is_empty() {
            return index;
        }

        let ml = 1.0 / (m as f64).ln();
        let mut rng_state = 0x123456789abcdef0u64;

        for (i, vec) in vectors.iter().enumerate() {
            // Pseudo-random level generation
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = ((rng_state >> 11) as f64 + 1.0) / (9007199254740992.0);
            let node_level = (-r.ln() * ml).floor() as usize;

            index.insert_node(i, vec, node_level, vectors);
        }

        index
    }

    fn insert_node(
        &mut self,
        node_id: usize,
        vector: &[f32],
        node_level: usize,
        all_vectors: &[Vec<f32>],
    ) {
        let mut node_neighbors = vec![Vec::new(); node_level + 1];

        let Some(mut curr_obj) = self.entry_point else {
            self.entry_point = Some(node_id);
            self.max_level = node_level;
            self.nodes.push(HnswNode {
                level: node_level,
                neighbors: node_neighbors,
            });
            return;
        };

        let curr_max_level = self.max_level;
        let mut curr_dist = cosine_distance(vector, &all_vectors[curr_obj]);

        // Greedily navigate down to min(node_level, curr_max_level) + 1
        for l in (node_level + 1..=curr_max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                if l < self.nodes[curr_obj].neighbors.len() {
                    for &neighbor in &self.nodes[curr_obj].neighbors[l] {
                        let dist = cosine_distance(vector, &all_vectors[neighbor]);
                        if dist < curr_dist {
                            curr_dist = dist;
                            curr_obj = neighbor;
                            changed = true;
                        }
                    }
                }
            }
        }

        // Connect at levels from min(node_level, curr_max_level) down to 0
        for l in (0..=node_level.min(curr_max_level)).rev() {
            let candidates =
                self.search_level(vector, curr_obj, self.ef_construction, l, all_vectors);
            let max_m = if l == 0 { self.m0 } else { self.m };
            let selected = select_neighbors(&candidates, max_m);

            node_neighbors[l] = selected.clone();

            for &neighbor in &selected {
                let neighbor_max_m = if l == 0 { self.m0 } else { self.m };
                if l < self.nodes[neighbor].neighbors.len() {
                    self.nodes[neighbor].neighbors[l].push(node_id);
                    if self.nodes[neighbor].neighbors[l].len() > neighbor_max_m {
                        let n_vec = &all_vectors[neighbor];
                        let n_candidates: Vec<Candidate> = self.nodes[neighbor].neighbors[l]
                            .iter()
                            .map(|&other| Candidate {
                                distance: cosine_distance(n_vec, &all_vectors[other]),
                                node_id: other,
                            })
                            .collect();
                        self.nodes[neighbor].neighbors[l] =
                            select_neighbors(&n_candidates, neighbor_max_m);
                    }
                }
            }

            if !candidates.is_empty() {
                curr_obj = candidates.first().unwrap().node_id;
            }
        }

        if node_level > curr_max_level {
            self.entry_point = Some(node_id);
            self.max_level = node_level;
        }

        self.nodes.push(HnswNode {
            level: node_level,
            neighbors: node_neighbors,
        });
    }

    fn search_level(
        &self,
        query: &[f32],
        entry: usize,
        ef: usize,
        level: usize,
        all_vectors: &[Vec<f32>],
    ) -> Vec<Candidate> {
        let mut visited = HashSet::new();
        visited.insert(entry);

        let entry_dist = cosine_distance(query, &all_vectors[entry]);
        let mut v_min = BinaryHeap::new();
        let mut w_max = BinaryHeap::new();

        v_min.push(MinCandidate {
            distance: entry_dist,
            node_id: entry,
        });
        w_max.push(Candidate {
            distance: entry_dist,
            node_id: entry,
        });

        while let Some(curr) = v_min.pop() {
            let furthest = w_max.peek().unwrap().distance;
            if curr.distance > furthest {
                break;
            }

            if level < self.nodes[curr.node_id].neighbors.len() {
                for &neighbor in &self.nodes[curr.node_id].neighbors[level] {
                    if visited.insert(neighbor) {
                        let dist = cosine_distance(query, &all_vectors[neighbor]);
                        let furthest_w = w_max.peek().unwrap().distance;

                        if dist < furthest_w || w_max.len() < ef {
                            v_min.push(MinCandidate {
                                distance: dist,
                                node_id: neighbor,
                            });
                            w_max.push(Candidate {
                                distance: dist,
                                node_id: neighbor,
                            });
                            if w_max.len() > ef {
                                w_max.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut res: Vec<Candidate> = w_max.into_vec();
        res.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        res
    }

    /// Queries the HNSW index for the nearest `k` neighbors.
    pub fn search(&self, query: &[f32], k: usize, all_vectors: &[Vec<f32>]) -> Vec<(usize, f32)> {
        let Some(mut curr_obj) = self.entry_point else {
            return Vec::new();
        };

        if all_vectors.is_empty() {
            return Vec::new();
        }

        let mut curr_dist = cosine_distance(query, &all_vectors[curr_obj]);

        // Greedily navigate top levels down to 1
        for l in (1..=self.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                if l < self.nodes[curr_obj].neighbors.len() {
                    for &neighbor in &self.nodes[curr_obj].neighbors[l] {
                        let dist = cosine_distance(query, &all_vectors[neighbor]);
                        if dist < curr_dist {
                            curr_dist = dist;
                            curr_obj = neighbor;
                            changed = true;
                        }
                    }
                }
            }
        }

        // Beam search at level 0
        let candidates = self.search_level(query, curr_obj, self.ef_search.max(k), 0, all_vectors);
        candidates
            .into_iter()
            .take(k)
            .map(|c| (c.node_id, 1.0 - c.distance)) // Convert back to similarity score
            .collect()
    }

    /// Serializes the HNSW index into a compact binary format.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(
            &(self.entry_point.map(|e| e as u32).unwrap_or(u32::MAX)).to_le_bytes(),
        );
        buf.extend_from_slice(&(self.max_level as u32).to_le_bytes());
        buf.extend_from_slice(&(self.m as u32).to_le_bytes());
        buf.extend_from_slice(&(self.m0 as u32).to_le_bytes());
        buf.extend_from_slice(&(self.nodes.len() as u32).to_le_bytes());

        for node in &self.nodes {
            buf.extend_from_slice(&(node.level as u32).to_le_bytes());
            buf.extend_from_slice(&(node.neighbors.len() as u32).to_le_bytes());
            for level_neighbors in &node.neighbors {
                buf.extend_from_slice(&(level_neighbors.len() as u32).to_le_bytes());
                for &neighbor in level_neighbors {
                    buf.extend_from_slice(&(neighbor as u32).to_le_bytes());
                }
            }
        }
        buf
    }

    /// Deserializes an HNSW index from a binary byte slice.
    pub fn deserialize(bytes: &[u8]) -> anyhow::Result<Self> {
        if bytes.len() < 20 {
            return Err(anyhow::anyhow!("invalid HNSW binary length"));
        }
        let mut cursor = 0usize;

        let ep_raw = read_u32(bytes, &mut cursor)?;
        let entry_point = if ep_raw == u32::MAX {
            None
        } else {
            Some(ep_raw as usize)
        };
        let max_level = read_u32(bytes, &mut cursor)? as usize;
        let m = read_u32(bytes, &mut cursor)? as usize;
        let m0 = read_u32(bytes, &mut cursor)? as usize;
        let node_count = read_u32(bytes, &mut cursor)? as usize;

        let mut nodes = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            let level = read_u32(bytes, &mut cursor)? as usize;
            let level_count = read_u32(bytes, &mut cursor)? as usize;
            let mut neighbors = Vec::with_capacity(level_count);
            for _ in 0..level_count {
                let n_count = read_u32(bytes, &mut cursor)? as usize;
                let mut level_n = Vec::with_capacity(n_count);
                for _ in 0..n_count {
                    level_n.push(read_u32(bytes, &mut cursor)? as usize);
                }
                neighbors.push(level_n);
            }
            nodes.push(HnswNode { level, neighbors });
        }

        Ok(Self {
            entry_point,
            max_level,
            m,
            m0,
            ef_construction: 64,
            ef_search: 32,
            nodes,
        })
    }
}

fn select_neighbors(candidates: &[Candidate], max_m: usize) -> Vec<usize> {
    candidates.iter().take(max_m).map(|c| c.node_id).collect()
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> anyhow::Result<u32> {
    let end = *cursor + 4;
    let slice = bytes
        .get(*cursor..end)
        .ok_or_else(|| anyhow::anyhow!("unexpected EOF in HNSW binary"))?;
    *cursor = end;
    let array: [u8; 4] = slice.try_into().map_err(|_| anyhow::anyhow!("bad u32"))?;
    Ok(u32::from_le_bytes(array))
}
