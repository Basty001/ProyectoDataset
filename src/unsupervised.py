"""
Unsupervised Learning Module.
Applies PCA for dimensionality reduction and three clustering algorithms
(KMeans, DBSCAN, Agglomerative) to explore structure in the feature space.
All functions expect a scaled numeric feature matrix (X_train from preprocessing).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage

RANDOM_STATE = 42
PLOTS_DIR = Path("results/plots")
METRICS_DIR = Path("results/metrics")


# ─────────────────────────────────────────────────────────────────────────────
# PCA
# ─────────────────────────────────────────────────────────────────────────────

def run_pca(X: pd.DataFrame, n_components: int = None, variance_threshold: float = 0.95) -> tuple:
    """
    Fits PCA on X and returns the reduced matrix plus the fitted model.

    If n_components is None, the number of components is chosen automatically
    to explain at least ``variance_threshold`` of total variance.

    Parameters
    ----------
    X : pd.DataFrame
        Scaled feature matrix (numeric only).
    n_components : int, optional
        Explicit number of components. Overrides variance_threshold.
    variance_threshold : float
        Minimum cumulative explained variance when n_components is None.

    Returns
    -------
    X_pca : np.ndarray  — transformed data
    pca   : PCA         — fitted model
    n_keep: int         — number of components retained
    """
    # Select only numeric columns to be safe
    X_num = X.select_dtypes(include=[np.number])

    if n_components is None:
        # Fit full PCA first to find elbow
        pca_full = PCA(random_state=RANDOM_STATE)
        pca_full.fit(X_num)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_keep = int(np.searchsorted(cumvar, variance_threshold)) + 1
    else:
        n_keep = n_components

    pca = PCA(n_components=n_keep, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_num)

    print(f">> PCA: {X_num.shape[1]} features → {n_keep} components "
          f"({pca.explained_variance_ratio_.sum()*100:.1f}% variance retained)")
    return X_pca, pca, n_keep


def plot_pca_variance(X: pd.DataFrame, save: bool = True) -> None:
    """
    Plots the cumulative explained variance curve.

    Parameters
    ----------
    X : pd.DataFrame
        Scaled feature matrix.
    save : bool
        If True, saves the figure to results/plots/.
    """
    X_num = X.select_dtypes(include=[np.number])
    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X_num)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(1, len(cumvar) + 1), cumvar, marker="o", markersize=3, linewidth=1.5)
    ax.axhline(95, color="red", linestyle="--", label="95% threshold")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("PCA – Cumulative Explained Variance")
    ax.legend()
    plt.tight_layout()

    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / "pca_variance.png", dpi=150)
        print(f"  Guardado: Saved: {PLOTS_DIR}/pca_variance.png")
    plt.close(fig)


def plot_pca_2d(X_pca: np.ndarray, labels: np.ndarray = None, title: str = "PCA 2D Projection", save: bool = True) -> None:
    """
    Scatter plot of the first two principal components, coloured by labels if given.

    Parameters
    ----------
    X_pca : np.ndarray
        PCA-reduced data (at least 2 columns).
    labels : np.ndarray, optional
        Cluster or category labels for colouring.
    title : str
        Plot title.
    save : bool
        If True, saves the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=labels if labels is not None else "steelblue",
        cmap="tab10", alpha=0.5, s=10
    )
    if labels is not None:
        plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    plt.tight_layout()

    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fname = title.lower().replace(" ", "_") + ".png"
        fig.savefig(PLOTS_DIR / fname, dpi=150)
        print(f"  Guardado: Saved: {PLOTS_DIR}/{fname}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# KMeans
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_k(X_pca: np.ndarray, k_range: range = range(2, 11)) -> int:
    """
    Uses the Elbow Method + Silhouette Score to suggest the optimal k.

    Parameters
    ----------
    X_pca : np.ndarray
        PCA-reduced feature matrix.
    k_range : range
        Range of k values to evaluate.

    Returns
    -------
    int
        Recommended k based on highest silhouette score.
    """
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_pca, labels, sample_size=2000, random_state=RANDOM_STATE))

    best_k = list(k_range)[int(np.argmax(silhouettes))]
    print(f">> Optimal k by silhouette: {best_k}  (silhouette={max(silhouettes):.4f})")

    # Plot elbow + silhouette side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(list(k_range), inertias, marker="o")
    ax1.set_title("KMeans – Elbow Method")
    ax1.set_xlabel("k"); ax1.set_ylabel("Inertia")

    ax2.plot(list(k_range), silhouettes, marker="o", color="green")
    ax2.axvline(best_k, color="red", linestyle="--", label=f"Best k={best_k}")
    ax2.set_title("KMeans – Silhouette Score")
    ax2.set_xlabel("k"); ax2.set_ylabel("Silhouette")
    ax2.legend()

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / "kmeans_elbow_silhouette.png", dpi=150)
    plt.close(fig)
    print(f"  Guardado: Saved: {PLOTS_DIR}/kmeans_elbow_silhouette.png")
    return best_k


def run_kmeans(X_pca: np.ndarray, k: int) -> np.ndarray:
    """
    Fits KMeans with k clusters and returns labels.

    Parameters
    ----------
    X_pca : np.ndarray
        PCA-reduced feature matrix.
    k : int
        Number of clusters.

    Returns
    -------
    np.ndarray
        Cluster labels.
    """
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels, sample_size=2000, random_state=RANDOM_STATE)
    db  = davies_bouldin_score(X_pca, labels)
    print(f">> KMeans (k={k})  |  Silhouette={sil:.4f}  |  Davies-Bouldin={db:.4f}")
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# DBSCAN
# ─────────────────────────────────────────────────────────────────────────────

def run_dbscan(X_pca: np.ndarray, eps: float = 1.5, min_samples: int = 10) -> np.ndarray:
    """
    Fits DBSCAN and returns labels (-1 = noise).

    Parameters
    ----------
    X_pca : np.ndarray
        PCA-reduced feature matrix.
    eps : float
        Maximum distance between two samples to be considered neighbours.
    min_samples : int
        Minimum samples in a neighbourhood to form a core point.

    Returns
    -------
    np.ndarray
        Cluster labels (-1 for noise).
    """
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X_pca)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = np.sum(labels == -1)
    print(f">> DBSCAN (eps={eps}, min_samples={min_samples})  |  "
          f"Clusters={n_clusters}  |  Noise points={n_noise} ({100*n_noise/len(labels):.1f}%)")
    if n_clusters > 1:
        mask = labels != -1
        sil = silhouette_score(X_pca[mask], labels[mask], sample_size=min(2000, mask.sum()), random_state=RANDOM_STATE)
        print(f"   Silhouette (excl. noise): {sil:.4f}")
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Agglomerative (Hierarchical)
# ─────────────────────────────────────────────────────────────────────────────

def plot_dendrogram(X_pca: np.ndarray, sample_size: int = 300, save: bool = True) -> None:
    """
    Plots a truncated dendrogram using Ward linkage on a random subsample.

    Parameters
    ----------
    X_pca : np.ndarray
        PCA-reduced feature matrix.
    sample_size : int
        Number of rows to subsample (full dataset too large for dendrograms).
    save : bool
        If True, saves the figure.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_pca), size=min(sample_size, len(X_pca)), replace=False)
    X_sub = X_pca[idx]

    Z = linkage(X_sub, method="ward")
    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(Z, truncate_mode="lastp", p=20, ax=ax,
               leaf_rotation=90, leaf_font_size=9)
    ax.set_title(f"Hierarchical Clustering Dendrogram (sample n={len(X_sub)})")
    ax.set_xlabel("Sample index / Cluster size")
    ax.set_ylabel("Ward Distance")
    plt.tight_layout()

    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOTS_DIR / "dendrogram.png", dpi=150)
        print(f"  Guardado: Saved: {PLOTS_DIR}/dendrogram.png")
    plt.close(fig)


def run_agglomerative(X_pca: np.ndarray, k: int) -> np.ndarray:
    """
    Fits Agglomerative (Ward linkage) clustering.

    Parameters
    ----------
    X_pca : np.ndarray
        PCA-reduced feature matrix.
    k : int
        Number of clusters.

    Returns
    -------
    np.ndarray
        Cluster labels.
    """
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = agg.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels, sample_size=2000, random_state=RANDOM_STATE)
    db  = davies_bouldin_score(X_pca, labels)
    print(f">> Agglomerative (k={k}, Ward)  |  Silhouette={sil:.4f}  |  Davies-Bouldin={db:.4f}")
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def clustering_summary(X_pca: np.ndarray, labels_dict: dict) -> pd.DataFrame:
    """
    Builds a comparison table of clustering metrics.

    Parameters
    ----------
    X_pca : np.ndarray
        PCA-reduced feature matrix.
    labels_dict : dict
        {algorithm_name: labels_array}

    Returns
    -------
    pd.DataFrame
        Table with Silhouette and Davies-Bouldin per algorithm.
    """
    rows = []
    for name, labels in labels_dict.items():
        mask = labels != -1
        n_clusters = len(set(labels[mask]))
        if n_clusters > 1 and mask.sum() > 1:
            sil = silhouette_score(X_pca[mask], labels[mask],
                                   sample_size=min(2000, mask.sum()), random_state=RANDOM_STATE)
            db  = davies_bouldin_score(X_pca[mask], labels[mask])
        else:
            sil, db = np.nan, np.nan
        rows.append({"Algorithm": name, "n_clusters": n_clusters,
                     "Silhouette ↑": round(sil, 4), "Davies-Bouldin ↓": round(db, 4)})
    df_summary = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(METRICS_DIR / "clustering_metrics.csv", index=False)
    print(f"\n>> Clustering Summary:\n{df_summary.to_string(index=False)}")
    return df_summary


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def run_unsupervised(X_train: pd.DataFrame) -> dict:
    """
    Orchestrates the full unsupervised analysis pipeline.

    Steps
    -----
    1. PCA (95 % variance retained).
    2. Elbow + Silhouette to choose optimal k.
    3. KMeans, DBSCAN, Agglomerative with the chosen k.
    4. 2-D PCA scatter plots for each algorithm.
    5. Dendrogram (subsample).
    6. Metrics summary CSV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix (already scaled by preprocessing pipeline).

    Returns
    -------
    dict
        {'X_pca': np.ndarray, 'labels': {'kmeans': ..., 'dbscan': ..., 'agg': ...}}
    """
    print("\n" + "="*60)
    print("  UNSUPERVISED LEARNING")
    print("="*60)

    # 1. PCA
    plot_pca_variance(X_train)
    X_pca, pca, n_comp = run_pca(X_train)
    plot_pca_2d(X_pca, title="PCA 2D Projection (no labels)")

    # 2. Optimal k
    best_k = find_optimal_k(X_pca)

    # 3. Clustering
    labels_km  = run_kmeans(X_pca, k=best_k)
    labels_db  = run_dbscan(X_pca, eps=1.5, min_samples=10)
    labels_agg = run_agglomerative(X_pca, k=best_k)

    # 4. 2-D projections coloured by cluster
    plot_pca_2d(X_pca, labels_km,  title="PCA 2D – KMeans Clusters")
    plot_pca_2d(X_pca, labels_db,  title="PCA 2D – DBSCAN Clusters")
    plot_pca_2d(X_pca, labels_agg, title="PCA 2D – Agglomerative Clusters")

    # 5. Dendrogram
    plot_dendrogram(X_pca)

    # 6. Summary
    clustering_summary(X_pca, {"KMeans": labels_km, "DBSCAN": labels_db, "Agglomerative": labels_agg})

    return {
        "X_pca": X_pca,
        "pca_model": pca,
        "labels": {"kmeans": labels_km, "dbscan": labels_db, "agglomerative": labels_agg},
        "best_k": best_k,
    }


if __name__ == "__main__":
    import pandas as pd
    X_train = pd.read_csv("data/processed/X_train.csv")
    run_unsupervised(X_train)
