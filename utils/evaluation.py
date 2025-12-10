import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr


def evaluate_marker_monotonicity_general(
    X_syn,
    traj_coord,
    gene_list,
    marker_sets,
    expected_directions=None,
):
    """
    General marker monotonicity evaluation for arbitrary trajectories.

    Parameters
    ----------
    X_syn : np.ndarray
        Synthetic expression matrix of shape (n_steps, n_genes).
    traj_coord : array-like
        1D array of length n_steps. Could be alpha in [0,1], pseudotime, etc.
    gene_list : list of str
        Gene names corresponding to columns of X_syn.
    marker_sets : dict
        Dict mapping group/state name -> list of marker genes.
        Example:
            {
                "Epithelial": ["CDH1", "EPCAM"],
                "Mesenchymal": ["VIM", "FN1"],
                "Basal": ["KRT5", "KRT14"],
            }
    expected_directions : dict or None
        Optional dict specifying expected direction of change for each group
        or for each gene.
        Two supported formats:

        1) Per-group:
            {
                "Epithelial": -1,   # expected to go DOWN along traj_coord
                "Mesenchymal": 1,   # expected to go UP
                "Basal": 0,         # no expectation / don't check
            }

        2) Per-gene:
            {
                "CDH1": -1, "EPCAM": -1,
                "VIM": 1, "FN1": 1,
            }

        Values should be in {-1, 0, 1}. If None, no direction check is done.

    Returns
    -------
    df : pd.DataFrame
        Columns:
            - group: marker set name
            - gene: gene name
            - rho: Spearman correlation between traj_coord and gene expression
            - pval: p-value of Spearman test
            - expected_dir: expected direction (-1, 0, 1 or NaN)
            - sign_ok: True if sign(rho) matches expected_dir (when expected_dir != 0)
    summary : pd.DataFrame
        One row per group, with aggregate stats:
            - group
            - n_genes
            - mean_rho
            - median_rho
            - frac_sign_ok (if expected_directions given)
    """
    traj_coord = np.asarray(traj_coord).ravel()
    assert X_syn.shape[0] == traj_coord.shape[0], "X_syn and traj_coord length mismatch"

    gene2idx = {g: i for i, g in enumerate(gene_list)}

    rows = []

    # Helper to get expected direction for a given (group, gene)
    def get_expected_dir(group, gene):
        if expected_directions is None:
            return np.nan
        # per-gene spec takes precedence
        if isinstance(expected_directions, dict):
            if gene in expected_directions:
                return expected_directions[gene]
            if group in expected_directions:
                return expected_directions[group]
        return np.nan

    for group, markers in marker_sets.items():
        for g in markers:
            if g not in gene2idx:
                continue
            idx = gene2idx[g]
            expr = X_syn[:, idx]

            # Spearman correlation
            rho, p = spearmanr(traj_coord, expr)
            exp_dir = get_expected_dir(group, g)

            if np.isnan(exp_dir) or exp_dir == 0:
                sign_ok = np.nan
            else:
                sign_ok = np.sign(rho) == np.sign(exp_dir)

            rows.append(
                {
                    "group": group,
                    "gene": g,
                    "rho": rho,
                    "pval": p,
                    "expected_dir": exp_dir,
                    "sign_ok": sign_ok,
                }
            )

    df = pd.DataFrame(rows)

    # Aggregate per group
    summary_rows = []
    for group, sub in df.groupby("group"):
        n = len(sub)
        mean_rho = sub["rho"].mean() if n > 0 else np.nan
        median_rho = sub["rho"].median() if n > 0 else np.nan

        if "sign_ok" in sub and sub["sign_ok"].notna().any():
            frac_sign_ok = sub["sign_ok"].dropna().mean()
        else:
            frac_sign_ok = np.nan

        summary_rows.append(
            {
                "group": group,
                "n_genes": n,
                "mean_rho": mean_rho,
                "median_rho": median_rho,
                "frac_sign_ok": frac_sign_ok,
            }
        )

    summary = pd.DataFrame(summary_rows)

    return df, summary


def evaluate_pseudotime_distance_error(
    X_syn,
    alphas,
    adata,
    gene_list,
    real_pseudotime_key="dpt_pseudotime",
    n_neighbors=1,
    metric="euclidean",
):
    """
    For each synthetic cell, find nearest real cell in gene space and
    compare synthetic alpha to real pseudotime.

    Returns:
        df: DataFrame with alpha, nearest_t, delta_t, dist
        summary: dict of aggregate metrics
    """

    # subset real data to same genes & densify
    common_genes = [g for g in gene_list if g in adata.var_names]
    if len(common_genes) != len(gene_list):
        # align X_syn to adata.var order for these genes
        # build mapping from gene_list index -> adata.var index
        gene2col_real = {g: i for i, g in enumerate(adata.var_names)}
        cols_real = [gene2col_real[g] for g in common_genes]
        X_real = adata.X[:, cols_real]
        X_syn_use = X_syn[:, [gene_list.index(g) for g in common_genes]]
    else:
        X_real = adata.X
        X_syn_use = X_syn

    if not isinstance(X_real, np.ndarray):
        X_real = X_real.toarray()
    X_real = X_real.astype(np.float32)
    X_syn_use = X_syn_use.astype(np.float32)

    # get real pseudotime
    t_real = adata.obs[real_pseudotime_key].to_numpy().astype(np.float32)

    # fit kNN on real cells
    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=metric,
    )
    nn.fit(X_real)

    dists, indices = nn.kneighbors(X_syn_use, return_distance=True)
    # use nearest neighbor only
    nearest_idx = indices[:, 0]
    nearest_dist = dists[:, 0]
    nearest_t = t_real[nearest_idx]

    delta_t = np.abs(nearest_t - alphas)

    df = pd.DataFrame({
        "alpha": alphas,
        "nearest_pseudotime": nearest_t,
        "delta_t": delta_t,
        "dist_to_nearest": nearest_dist,
    })

    summary = {
        "delta_t_mean": float(delta_t.mean()),
        "delta_t_median": float(np.median(delta_t)),
        "delta_t_max": float(delta_t.max()),
        "dist_mean": float(nearest_dist.mean()),
        "dist_median": float(np.median(nearest_dist)),
        "dist_max": float(nearest_dist.max()),
    }

    return df, summary

def evaluate_distance_to_manifold_smoothness(
    X_syn,
    adata,
    gene_list,
    n_neighbors=1,
    metric="euclidean",
):
    """
    Compute distance from each synthetic point to the nearest real cell
    and basic smoothness statistics on this distance along the trajectory.
    """

    # subset real data to same genes & densify
    common_genes = [g for g in gene_list if g in adata.var_names]
    if len(common_genes) != len(gene_list):
        gene2col_real = {g: i for i, g in enumerate(adata.var_names)}
        cols_real = [gene2col_real[g] for g in common_genes]
        X_real = adata.X[:, cols_real]
        X_syn_use = X_syn[:, [gene_list.index(g) for g in common_genes]]
    else:
        X_real = adata.X
        X_syn_use = X_syn

    if not isinstance(X_real, np.ndarray):
        X_real = X_real.toarray()
    X_real = X_real.astype(np.float32)
    X_syn_use = X_syn_use.astype(np.float32)

    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=metric,
    )
    nn.fit(X_real)

    dists, _ = nn.kneighbors(X_syn_use, return_distance=True)
    nearest_dist = dists[:, 0]

    # smoothness: how much the distance changes between consecutive steps
    diffs = np.diff(nearest_dist)
    abs_diffs = np.abs(diffs)

    summary = {
        "dist_mean": float(nearest_dist.mean()),
        "dist_median": float(np.median(nearest_dist)),
        "dist_max": float(nearest_dist.max()),
        "smooth_mean_abs_delta": float(abs_diffs.mean()),
        "smooth_max_abs_delta": float(abs_diffs.max()),
    }

    return nearest_dist, summary

