"""
Patent Company Graph - NLP Similarity & Clustering Pipeline
============================================================
Two embedding approaches compared:
  1. TF-IDF (baseline) — bag-of-words, keyword-based
  2. Sentence-BERT (SBERT) — semantic embeddings

Both are evaluated with:
  - Cosine similarity matrix
  - K-Means + Hierarchical clustering
  - Silhouette score comparison
  - Company similarity graph (NetworkX)
  - LDA topic modeling

Setup:
  pip install pandas scikit-learn networkx matplotlib seaborn scipy sentence-transformers
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path("data")
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)

INPUT_FILE = DATA_DIR / "epo_patents_translated.csv"
BEST_K = 5          # number of clusters (matches 5 IPC sections)
SIM_THRESHOLD = 0.3 # minimum similarity for graph edges (TF-IDF)
SBERT_THRESHOLD = 0.85  # higher threshold for SBERT (similarities are higher overall)

# ============================================================
# 1. LOAD & PREPARE DATA
# ============================================================

print("=" * 60)
print("Patent Company Graph - NLP Similarity Pipeline")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)
print(f"\nLoaded {len(df)} patents from {INPUT_FILE}")

# Keep only patents with abstracts
df = df[df["abstract"].notna() & (df["abstract"].astype(str).str.len() > 50)].copy()
print(f"After filtering (abstract > 50 chars): {len(df)} patents")
print(f"Companies: {df['company'].nunique()}")

companies = sorted(df["company"].unique())


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def build_company_vectors(patent_matrix, df, companies):
    """Average patent vectors per company and L2 normalize."""
    vectors = np.zeros((len(companies), patent_matrix.shape[1]))
    for i, company in enumerate(companies):
        mask = df["company"] == company
        vectors[i] = patent_matrix[mask.values].mean(axis=0)
    vectors = normalize(vectors)
    return vectors


def compute_similarity(company_vectors, companies):
    """Compute cosine similarity matrix."""
    sim_matrix = cosine_similarity(company_vectors)
    sim_df = pd.DataFrame(sim_matrix, index=companies, columns=companies)
    return sim_matrix, sim_df


def run_clustering(company_vectors, companies, k):
    """Run K-Means and Hierarchical clustering."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_km = km.fit_predict(company_vectors)

    agg = AgglomerativeClustering(n_clusters=k)
    labels_agg = agg.fit_predict(company_vectors)

    return labels_km, labels_agg


def print_clusters(labels, companies, method_name, k):
    """Print cluster assignments."""
    print(f"\n{method_name} (k={k}) cluster assignments:")
    for cid in range(k):
        members = [companies[i] for i in range(len(companies)) if labels[i] == cid]
        print(f"  Cluster {cid}: {', '.join(members)}")


def print_top_pairs(sim_matrix, companies, n=15):
    """Print most and least similar company pairs."""
    pairs = []
    for i in range(len(companies)):
        for j in range(i + 1, len(companies)):
            pairs.append((companies[i], companies[j], sim_matrix[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"\nTop {n} most similar company pairs:")
    for c1, c2, sim in pairs[:n]:
        print(f"  {c1:<22s} <-> {c2:<22s}  similarity: {sim:.3f}")

    print(f"\nBottom 5 least similar:")
    for c1, c2, sim in pairs[-5:]:
        print(f"  {c1:<22s} <-> {c2:<22s}  similarity: {sim:.3f}")


# ============================================================
# 2. TF-IDF BASELINE
# ============================================================

print(f"\n{'='*60}")
print("METHOD 1: TF-IDF (Baseline)")
print(f"{'='*60}")

tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    min_df=3,
    max_df=0.85,
    ngram_range=(1, 2),
    sublinear_tf=True,
)

tfidf_matrix = tfidf.fit_transform(df["abstract"].astype(str))
print(f"\nTF-IDF matrix shape: {tfidf_matrix.shape}")

feature_names = tfidf.get_feature_names_out()
mean_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
top_idx = mean_tfidf.argsort()[-15:][::-1]
print(f"\nTop 15 terms (by mean TF-IDF):")
for i, idx in enumerate(top_idx, 1):
    print(f"  {i:2d}. {feature_names[idx]:<30s} ({mean_tfidf[idx]:.4f})")

# Company vectors
tfidf_company = build_company_vectors(tfidf_matrix.toarray(), df, companies)
print(f"Company vectors: {tfidf_company.shape}")

# Similarity
tfidf_sim, tfidf_sim_df = compute_similarity(tfidf_company, companies)
tfidf_sim_df.to_csv(DATA_DIR / "company_similarity_tfidf.csv")
print(f"Saved: data/company_similarity_tfidf.csv")
print_top_pairs(tfidf_sim, companies)

# Clustering
tfidf_km, tfidf_agg = run_clustering(tfidf_company, companies, BEST_K)
print_clusters(tfidf_km, companies, "K-Means (TF-IDF)", BEST_K)

# Silhouette score
tfidf_sil = silhouette_score(tfidf_company, tfidf_km)
print(f"\nSilhouette score (TF-IDF, k={BEST_K}): {tfidf_sil:.4f}")

# Top terms per company
print(f"\n--- Top 5 distinctive terms per company ---")
for i, company in enumerate(companies):
    top_term_idx = tfidf_company[i].argsort()[-5:][::-1]
    terms = [feature_names[j] for j in top_term_idx]
    print(f"  {company:<22s}: {', '.join(terms)}")


# ============================================================
# 3. SENTENCE-BERT EMBEDDINGS
# ============================================================

print(f"\n{'='*60}")
print("METHOD 2: Sentence-BERT (Semantic Embeddings)")
print(f"{'='*60}")

from sentence_transformers import SentenceTransformer

# Load model
model_name = "all-MiniLM-L6-v2"
print(f"\nLoading SBERT model: {model_name}")
sbert_model = SentenceTransformer(model_name)

# Encode all abstracts
print("Encoding patent abstracts...")
abstracts = df["abstract"].astype(str).tolist()
sbert_embeddings = sbert_model.encode(abstracts, batch_size=64, show_progress_bar=True)
print(f"SBERT embeddings shape: {sbert_embeddings.shape}")

# Company vectors
sbert_company = build_company_vectors(sbert_embeddings, df, companies)
print(f"Company vectors: {sbert_company.shape}")

# Similarity
sbert_sim, sbert_sim_df = compute_similarity(sbert_company, companies)
sbert_sim_df.to_csv(DATA_DIR / "company_similarity_sbert.csv")
print(f"Saved: data/company_similarity_sbert.csv")
print_top_pairs(sbert_sim, companies)

# Clustering
sbert_km, sbert_agg = run_clustering(sbert_company, companies, BEST_K)
print_clusters(sbert_km, companies, "K-Means (SBERT)", BEST_K)

# Silhouette score
sbert_sil = silhouette_score(sbert_company, sbert_km)
print(f"\nSilhouette score (SBERT, k={BEST_K}): {sbert_sil:.4f}")


# ============================================================
# 4. COMPARISON: TF-IDF vs SBERT
# ============================================================

print(f"\n{'='*60}")
print("COMPARISON: TF-IDF vs SBERT")
print(f"{'='*60}")

print(f"\n{'Metric':<35s} {'TF-IDF':>10s} {'SBERT':>10s}")
print("-" * 57)
print(f"{'Embedding dimensions':<35s} {tfidf_company.shape[1]:>10d} {sbert_company.shape[1]:>10d}")
print(f"{'Silhouette score (k=5)':<35s} {tfidf_sil:>10.4f} {sbert_sil:>10.4f}")

# Mean off-diagonal similarity
tfidf_mean_sim = tfidf_sim[np.triu_indices_from(tfidf_sim, k=1)].mean()
sbert_mean_sim = sbert_sim[np.triu_indices_from(sbert_sim, k=1)].mean()
print(f"{'Mean pairwise similarity':<35s} {tfidf_mean_sim:>10.4f} {sbert_mean_sim:>10.4f}")

tfidf_max_sim = tfidf_sim[np.triu_indices_from(tfidf_sim, k=1)].max()
sbert_max_sim = sbert_sim[np.triu_indices_from(sbert_sim, k=1)].max()
print(f"{'Max similarity (off-diagonal)':<35s} {tfidf_max_sim:>10.4f} {sbert_max_sim:>10.4f}")

tfidf_min_sim = tfidf_sim[np.triu_indices_from(tfidf_sim, k=1)].min()
sbert_min_sim = sbert_sim[np.triu_indices_from(sbert_sim, k=1)].min()
print(f"{'Min similarity (off-diagonal)':<35s} {tfidf_min_sim:>10.4f} {sbert_min_sim:>10.4f}")

# Winner
if sbert_sil > tfidf_sil:
    print(f"\n>> SBERT produces better-separated clusters (silhouette: {sbert_sil:.4f} vs {tfidf_sil:.4f})")
else:
    print(f"\n>> TF-IDF produces better-separated clusters (silhouette: {tfidf_sil:.4f} vs {sbert_sil:.4f})")


# ============================================================
# 5. PLOTS (using SBERT as primary, TF-IDF for comparison)
# ============================================================

print(f"\n--- Generating plots ---")

# --- Plot 7: TF-IDF similarity heatmap ---
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(tfidf_sim_df, annot=False, cmap="YlOrRd", ax=ax,
            linewidths=0.3, vmin=0, vmax=1)
ax.set_title("Company Similarity Matrix — TF-IDF (Baseline)", fontsize=14)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
fig.savefig(PLOT_DIR / "07_similarity_heatmap_tfidf.png", dpi=150)
print(f"Saved: plots/07_similarity_heatmap_tfidf.png")

# --- Plot 8: SBERT similarity heatmap ---
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(sbert_sim_df, annot=False, cmap="YlOrRd", ax=ax,
            linewidths=0.3, vmin=0, vmax=1)
ax.set_title("Company Similarity Matrix — Sentence-BERT", fontsize=14)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
fig.savefig(PLOT_DIR / "08_similarity_heatmap_sbert.png", dpi=150)
print(f"Saved: plots/08_similarity_heatmap_sbert.png")

# --- Plot 9: Silhouette comparison bar chart ---
fig, ax = plt.subplots(figsize=(8, 5))
methods = ["TF-IDF", "SBERT"]
scores = [tfidf_sil, sbert_sil]
colors = ["#94A3B8", "#2A9D8F"]
bars = ax.bar(methods, scores, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{score:.4f}", ha="center", fontsize=12, fontweight="bold")
ax.set_ylabel("Silhouette Score")
ax.set_title("Cluster Quality Comparison: TF-IDF vs SBERT (k=5)")
ax.set_ylim(0, max(scores) * 1.3)
plt.tight_layout()
fig.savefig(PLOT_DIR / "09_silhouette_comparison.png", dpi=150)
print(f"Saved: plots/09_silhouette_comparison.png")

# --- Plot 10: K-Means elbow plot (SBERT) ---
inertias = []
sil_scores = []
K_range = range(2, 11)
for k in K_range:
    km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_tmp = km_tmp.fit_predict(sbert_company)
    inertias.append(km_tmp.inertia_)
    sil_scores.append(silhouette_score(sbert_company, labels_tmp))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(K_range, inertias, "bo-")
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia")
ax1.set_title("Elbow Plot (SBERT)")

ax2.plot(K_range, sil_scores, "ro-")
ax2.set_xlabel("Number of Clusters (k)")
ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Score vs k (SBERT)")
plt.tight_layout()
fig.savefig(PLOT_DIR / "10_elbow_silhouette_sbert.png", dpi=150)
print(f"Saved: plots/10_elbow_silhouette_sbert.png")

# --- Plot 11: Dendrogram (SBERT) ---
linkage_matrix = linkage(sbert_company, method="ward", metric="euclidean")

fig, ax = plt.subplots(figsize=(16, 8))
dendrogram(linkage_matrix, labels=companies, leaf_rotation=45, leaf_font_size=10, ax=ax)
ax.set_title("Hierarchical Clustering — Sentence-BERT (Ward Linkage)", fontsize=14)
ax.set_ylabel("Distance")
plt.tight_layout()
fig.savefig(PLOT_DIR / "11_dendrogram_sbert.png", dpi=150)
print(f"Saved: plots/11_dendrogram_sbert.png")

# --- Plot 12: t-SNE visualization (SBERT) ---
tsne = TSNE(n_components=2, random_state=42, perplexity=min(10, len(companies)-1))
coords_2d = tsne.fit_transform(sbert_company)

fig, ax = plt.subplots(figsize=(14, 10))
scatter = ax.scatter(
    coords_2d[:, 0], coords_2d[:, 1],
    c=sbert_km, cmap="Set1", s=200, edgecolors="black", linewidth=0.5
)
for i, company in enumerate(companies):
    ax.annotate(company, (coords_2d[i, 0], coords_2d[i, 1]),
                fontsize=9, fontweight="bold",
                xytext=(5, 5), textcoords="offset points")
ax.set_title("Company Similarity Map — SBERT (t-SNE + K-Means)", fontsize=14)
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
plt.colorbar(scatter, label="Cluster", ax=ax)
plt.tight_layout()
fig.savefig(PLOT_DIR / "12_tsne_sbert.png", dpi=150)
print(f"Saved: plots/12_tsne_sbert.png")


# ============================================================
# 6. COMPANY SIMILARITY GRAPH (using SBERT)
# ============================================================

print(f"\n--- Company Similarity Graph (SBERT) ---")

# Use the higher threshold for SBERT — with threshold 0.3 everything is connected
# because SBERT similarities are much higher overall (min ~0.38)
G = nx.Graph()
for i, company in enumerate(companies):
    G.add_node(company, cluster=int(sbert_km[i]))

for i in range(len(companies)):
    for j in range(i + 1, len(companies)):
        if sbert_sim[i, j] > SBERT_THRESHOLD:
            G.add_edge(companies[i], companies[j], weight=round(sbert_sim[i, j], 3))

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (threshold={SBERT_THRESHOLD})")
print(f"Density: {nx.density(G):.3f}")
print(f"Average clustering coefficient: {nx.average_clustering(G):.3f}")

degree_cent = nx.degree_centrality(G)
print(f"\nTop 10 by degree centrality:")
for company, cent in sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {company:<22s}: {cent:.3f} (degree: {G.degree(company)})")

between_cent = nx.betweenness_centrality(G)
print(f"\nTop 10 by betweenness centrality:")
for company, cent in sorted(between_cent.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {company:<22s}: {cent:.3f}")

nx.write_gexf(G, str(DATA_DIR / "company_graph_sbert.gexf"))
print(f"\nSaved: data/company_graph_sbert.gexf")

# --- Plot 13: Network graph (SBERT) ---
fig, ax = plt.subplots(figsize=(16, 12))
pos = nx.spring_layout(G, k=2, iterations=50, seed=42, weight="weight")
cluster_colors = [sbert_km[companies.index(n)] for n in G.nodes()]
node_sizes = [300 + G.degree(n) * 80 for n in G.nodes()]
edge_weights = [G[u][v]["weight"] * 2 for u, v in G.edges()]

nx.draw_networkx_nodes(G, pos, ax=ax, node_color=cluster_colors,
                       cmap=plt.cm.Set1, node_size=node_sizes,
                       edgecolors="black", linewidths=0.5)
nx.draw_networkx_edges(G, pos, ax=ax, width=edge_weights, alpha=0.3, edge_color="gray")
nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold")
ax.set_title(f"Company Similarity Network — SBERT (threshold={SBERT_THRESHOLD})", fontsize=14)
ax.axis("off")
plt.tight_layout()
fig.savefig(PLOT_DIR / "13_company_graph_sbert.png", dpi=150)
print(f"Saved: plots/13_company_graph_sbert.png")


# ============================================================
# 7. TOPIC MODELING (LDA)
# ============================================================

print(f"\n--- Topic Modeling (LDA) ---")

N_TOPICS = 8

count_vec = CountVectorizer(
    max_features=3000,
    stop_words="english",
    min_df=5,
    max_df=0.8,
    ngram_range=(1, 2),
)
count_matrix = count_vec.fit_transform(df["abstract"].astype(str))
count_features = count_vec.get_feature_names_out()

lda = LatentDirichletAllocation(
    n_components=N_TOPICS,
    random_state=42,
    max_iter=20,
    learning_method="online",
)
lda.fit(count_matrix)

print(f"\n{N_TOPICS} topics discovered:")
for topic_id, topic in enumerate(lda.components_):
    top_words = [count_features[i] for i in topic.argsort()[-10:][::-1]]
    print(f"  Topic {topic_id}: {', '.join(top_words)}")

topic_dist = lda.transform(count_matrix)
df["dominant_topic"] = topic_dist.argmax(axis=1)

topic_company = df.groupby(["company", "dominant_topic"]).size().unstack(fill_value=0)
topic_company.to_csv(DATA_DIR / "company_topic_distribution.csv")
print(f"\nSaved: data/company_topic_distribution.csv")

# --- Plot 14: Topic distribution heatmap ---
fig, ax = plt.subplots(figsize=(14, 10))
topic_company_norm = topic_company.div(topic_company.sum(axis=1), axis=0)
sns.heatmap(topic_company_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax, linewidths=0.5)
ax.set_title("Topic Distribution per Company (LDA, normalized)", fontsize=14)
ax.set_xlabel("Topic")
ax.set_ylabel("")
plt.tight_layout()
fig.savefig(PLOT_DIR / "14_topic_distribution.png", dpi=150)
print(f"Saved: plots/14_topic_distribution.png")


# ============================================================
# 8. SAVE FINAL RESULTS
# ============================================================

cluster_df = pd.DataFrame({
    "company": companies,
    "kmeans_cluster_tfidf": tfidf_km,
    "kmeans_cluster_sbert": sbert_km,
    "hierarchical_cluster_sbert": sbert_agg,
    "degree_centrality": [degree_cent.get(c, 0) for c in companies],
    "betweenness_centrality": [between_cent.get(c, 0) for c in companies],
})
cluster_df.to_csv(DATA_DIR / "company_clusters.csv", index=False)
print(f"\nSaved: data/company_clusters.csv")

# Also save the SBERT similarity as the primary one
sbert_sim_df.to_csv(DATA_DIR / "company_similarity_matrix.csv")
print(f"Saved: data/company_similarity_matrix.csv (SBERT)")

print(f"\n{'='*60}")
print("PIPELINE COMPLETE")
print(f"{'='*60}")
print(f"\nComparison summary:")
print(f"  TF-IDF silhouette:  {tfidf_sil:.4f}")
print(f"  SBERT silhouette:   {sbert_sil:.4f}")
print(f"\nOutputs:")
print(f"  data/company_similarity_tfidf.csv   - TF-IDF similarity matrix")
print(f"  data/company_similarity_sbert.csv   - SBERT similarity matrix")
print(f"  data/company_similarity_matrix.csv  - Primary similarity (SBERT)")
print(f"  data/company_clusters.csv           - Cluster assignments (both methods)")
print(f"  data/company_graph_sbert.gexf       - Network graph")
print(f"  data/company_topic_distribution.csv - LDA topics per company")
print(f"  plots/07-14                         - All visualizations")
