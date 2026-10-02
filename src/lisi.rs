//! Port of `../harmonypy/src/lisi.hpp` — the Local Inverse Simpson Index
//! (C++ implementation replacing the sklearn-based Python version).
//!
//! Simple kd-tree for k-nearest-neighbor search, binary search for the beta
//! perplexity parameter, Simpson's index over neighbor category probabilities.
//!
//! Index-based loops are intentional throughout: they mirror the C++
//! reference's flat-buffer (column-major / kd-tree) indexing directly.

#![allow(clippy::needless_range_loop)]

use std::f64;

/// Error type for LISI computation.
#[derive(Debug, Clone)]
pub struct LisiError(pub String);

impl std::fmt::Display for LisiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for LisiError {}

/// Simple kd-tree for k-nearest-neighbor search (port of `lisi::KDTree`).
struct KDTree {
    nodes: Vec<KDNode>,
    root: i32,
}

struct KDNode {
    split_dim: i32,
    split_val: f64,
    point_idx: i32, // -1 for internal nodes
    left: i32,
    right: i32, // child indices (-1 for none)
}

impl KDTree {
    /// `data` is N x d, row-major (point i, dim j at data[i * d + j]).
    fn new(data: &[f64], n_points: usize, n_dims: usize) -> Self {
        let indices: Vec<usize> = (0..n_points).collect();
        let mut nodes = Vec::with_capacity(2 * n_points);
        let root = build_kd(data, n_dims, &mut nodes, &indices, 0, n_points, 0);
        KDTree { nodes, root }
    }

    fn sq_dist(&self, data: &[f64], n_dims: usize, a: usize, b: usize) -> f64 {
        let mut d = 0.0;
        for j in 0..n_dims {
            let diff = data[a * n_dims + j] - data[b * n_dims + j];
            d += diff * diff;
        }
        d
    }

    /// Find k nearest neighbors for query point (excluding itself).
    fn knn(
        &self,
        data: &[f64],
        n_dims: usize,
        query_idx: usize,
        k: usize,
    ) -> (Vec<usize>, Vec<f64>) {
        // Max-heap of (distance, index).
        let mut heap: Vec<(f64, usize)> = Vec::with_capacity(k + 1);
        let mut max_dist = f64::MAX;
        self.knn_search(
            data,
            n_dims,
            self.root,
            query_idx,
            k,
            &mut heap,
            &mut max_dist,
        );

        // Sort by distance.
        heap.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let nn_indices: Vec<usize> = heap.iter().map(|&(_, i)| i).collect();
        let nn_dists: Vec<f64> = heap.iter().map(|&(d, _)| d.sqrt()).collect();
        (nn_indices, nn_dists)
    }

    #[allow(clippy::too_many_arguments)]
    fn knn_search(
        &self,
        data: &[f64],
        n_dims: usize,
        node_idx: i32,
        query_idx: usize,
        k: usize,
        heap: &mut Vec<(f64, usize)>,
        max_dist: &mut f64,
    ) {
        if node_idx < 0 {
            return;
        }
        let node = &self.nodes[node_idx as usize];

        let pt = node.point_idx as usize;
        if pt != query_idx {
            let d = self.sq_dist(data, n_dims, query_idx, pt);
            if heap.len() < k {
                heap.push((d, pt));
                heap_sort_up(heap);
                if heap.len() == k {
                    *max_dist = heap[0].0;
                }
            } else if d < *max_dist {
                // pop max, push new
                heap[0] = (d, pt);
                heap_sift_down(heap);
                *max_dist = heap[0].0;
            }
        }

        if node.left < 0 && node.right < 0 {
            return;
        }

        let diff = data[query_idx * n_dims + node.split_dim as usize] - node.split_val;
        let (near, far) = if diff <= 0.0 {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        };

        self.knn_search(data, n_dims, near, query_idx, k, heap, max_dist);
        if diff * diff < *max_dist || heap.len() < k {
            self.knn_search(data, n_dims, far, query_idx, k, heap, max_dist);
        }
    }
}

/// Port of `KDTree::build` — recursive kd-tree construction over
/// indices[begin..end) with `std::nth_element` partitioning by the
/// depth-th coordinate.
fn build_kd(
    data: &[f64],
    n_dims: usize,
    nodes: &mut Vec<KDNode>,
    indices: &[usize],
    begin: usize,
    end: usize,
    depth: usize,
) -> i32 {
    if begin >= end {
        return -1;
    }
    if end - begin == 1 {
        let idx = nodes.len() as i32;
        nodes.push(KDNode {
            split_dim: -1,
            split_val: 0.0,
            point_idx: indices[begin] as i32,
            left: -1,
            right: -1,
        });
        return idx;
    }
    let dim = depth % n_dims;
    let mid = begin + (end - begin) / 2;
    // Port of std::nth_element over indices[begin..end).
    let mut range: Vec<usize> = indices[begin..end].to_vec();
    range.select_nth_unstable_by(mid - begin, |&a, &b| {
        data[a * n_dims + dim]
            .partial_cmp(&data[b * n_dims + dim])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mid_idx = range[mid - begin];
    let mid_val = data[mid_idx * n_dims + dim];
    let idx = nodes.len() as i32;
    nodes.push(KDNode {
        split_dim: dim as i32,
        split_val: mid_val,
        point_idx: mid_idx as i32,
        left: -1,
        right: -1,
    });
    let mid_local = mid - begin;
    nodes[idx as usize].left = build_kd(data, n_dims, nodes, &range, 0, mid_local, depth + 1);
    nodes[idx as usize].right = build_kd(
        data,
        n_dims,
        nodes,
        &range,
        mid_local + 1,
        end - begin,
        depth + 1,
    );
    idx
}

/// Max-heap helpers porting std::push_heap / std::pop_heap on
/// (distance, index) pairs with (f64, usize) ordering.
fn heap_sort_up(heap: &mut [(f64, usize)]) {
    let mut i = heap.len() - 1;
    while i > 0 {
        let parent = (i - 1) / 2;
        if heap[i] > heap[parent] {
            heap.swap(i, parent);
            i = parent;
        } else {
            break;
        }
    }
}

fn heap_sift_down(heap: &mut [(f64, usize)]) {
    let n = heap.len();
    let mut i = 0;
    loop {
        let left = 2 * i + 1;
        let right = 2 * i + 2;
        let mut largest = i;
        if left < n && heap[left] > heap[largest] {
            largest = left;
        }
        if right < n && heap[right] > heap[largest] {
            largest = right;
        }
        if largest == i {
            break;
        }
        heap.swap(i, largest);
        i = largest;
    }
}

/// Port of `lisi::compute_simpson_one` — Simpson's index for one cell given
/// its neighbor distances and labels.
#[allow(clippy::too_many_arguments)]
fn compute_simpson_one(
    distances: &[f64],
    indices: &[usize],
    labels: &[usize],
    n_categories: usize,
    perplexity: f64,
    tol: f64,
) -> f64 {
    let n_neighbors = distances.len();
    let log_u = perplexity.ln();
    let mut beta = 1.0;
    let mut betamin = f64::NEG_INFINITY;
    let mut betamax = f64::INFINITY;

    let mut p = vec![0.0f64; n_neighbors];
    let mut h = 0.0f64;

    // Binary search for beta that gives the target perplexity.
    for _t in 0..50 {
        let mut p_sum = 0.0f64;
        let mut sum_dp = 0.0f64;
        for j in 0..n_neighbors {
            p[j] = (-distances[j] * beta).exp();
            p_sum += p[j];
            sum_dp += distances[j] * p[j];
        }
        if p_sum == 0.0 {
            h = 0.0;
            for v in p.iter_mut() {
                *v = 0.0;
            }
        } else {
            h = p_sum.ln() + beta * sum_dp / p_sum;
            for v in p.iter_mut() {
                *v /= p_sum;
            }
        }

        let hdiff = h - log_u;
        if hdiff.abs() < tol {
            break;
        }

        if hdiff > 0.0 {
            betamin = beta;
            beta = if betamax.is_finite() {
                (beta + betamax) / 2.0
            } else {
                beta * 2.0
            };
        } else {
            betamax = beta;
            beta = if betamin.is_finite() {
                (beta + betamin) / 2.0
            } else {
                beta / 2.0
            };
        }
    }

    if h == 0.0 {
        return -1.0;
    }

    // Simpson's index: sum of squared category probabilities.
    let mut cat_prob = vec![0.0f64; n_categories];
    for j in 0..n_neighbors {
        cat_prob[labels[indices[j]]] += p[j];
    }
    let mut simpson = 0.0;
    for &cp in &cat_prob {
        simpson += cp * cp;
    }
    simpson
}

/// Port of `lisi::compute_lisi_impl` — compute LISI for all cells.
///
/// `x` is N x d, row-major; `labels` is N integer labels (0-indexed).
pub fn compute_lisi_impl(
    x: &[f64],
    n_points: usize,
    n_dims: usize,
    labels: &[usize],
    n_categories: usize,
    perplexity: f64,
) -> Vec<f64> {
    let k = (perplexity * 3.0) as usize;
    let tree = KDTree::new(x, n_points, n_dims);

    let mut lisi_result = vec![0.0f64; n_points];
    for i in 0..n_points {
        let (nn_idx, nn_dist) = tree.knn(x, n_dims, i, k);
        let simpson =
            compute_simpson_one(&nn_dist, &nn_idx, labels, n_categories, perplexity, 1e-5);
        lisi_result[i] = if simpson > 0.0 { 1.0 / simpson } else { 0.0 };
    }
    lisi_result
}
