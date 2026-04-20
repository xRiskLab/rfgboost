use ndarray::ArrayView2;

pub const MAX_BINS: usize = 256;

pub struct HistogramData {
    pub bin_indices: Vec<Vec<u8>>,
    pub bin_edges: Vec<Vec<f64>>,
    pub n_bins: Vec<usize>,
}

impl HistogramData {
    pub fn build(x: &ArrayView2<f64>, n_samples: usize, n_features: usize) -> Self {
        let mut bin_indices = Vec::with_capacity(n_features);
        let mut bin_edges = Vec::with_capacity(n_features);
        let mut n_bins_vec = Vec::with_capacity(n_features);

        for feat in 0..n_features {
            let mut vals: Vec<f64> = (0..n_samples).map(|i| x[[i, feat]]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let n_unique = {
                let mut v = vals.clone();
                v.dedup();
                v.len()
            };
            let actual_bins = n_unique.min(MAX_BINS);

            let mut edges = Vec::with_capacity(actual_bins);
            if actual_bins <= 1 {
                edges.push(f64::INFINITY);
            } else {
                for b in 1..actual_bins {
                    let idx = (b * n_samples) / actual_bins;
                    let idx = idx.min(n_samples - 1);
                    let edge = vals[idx];
                    if edges.is_empty() || edge > *edges.last().unwrap() {
                        edges.push(edge);
                    }
                }
                edges.push(f64::INFINITY);
            }

            let nb = edges.len();

            let mut feat_bins = vec![0u8; n_samples];
            for i in 0..n_samples {
                let v = x[[i, feat]];
                let mut lo = 0;
                let mut hi = nb;
                while lo < hi {
                    let mid = (lo + hi) / 2;
                    if v > edges[mid] {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                feat_bins[i] = lo as u8;
            }

            bin_indices.push(feat_bins);
            bin_edges.push(edges);
            n_bins_vec.push(nb);
        }

        HistogramData {
            bin_indices,
            bin_edges,
            n_bins: n_bins_vec,
        }
    }
}
