import os
import sys
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import zipfile
import subprocess
from itertools import combinations
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
import imageio

# =========================================
# Step 0: Paths & Output Setup
# =========================================
DATA_DIR = "data"
RESULTS_DIR = "results"
RECOMMENDER_DIR = os.path.join(RESULTS_DIR, "recommender")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RECOMMENDER_DIR, exist_ok=True)

COMMUNITY_DIR = os.path.join(RESULTS_DIR, "community")
os.makedirs(COMMUNITY_DIR, exist_ok=True)

# Diffusion results directory for PageRank and related outputs
DIFFUSION_DIR = os.path.join(RESULTS_DIR, "diffusion")
os.makedirs(DIFFUSION_DIR, exist_ok=True)

AMAZON_DATA_PATH = os.path.join(DATA_DIR, "ratings_Electronics (1).csv")

# =========================================
# Step 1: Download the dataset from Kaggle (if needed)
# =========================================

def ensure_kaggle_credentials():
    """Ensure Kaggle credentials exist in one of two places; set env if local is used."""
    default_path = os.path.expanduser("~/.kaggle/kaggle.json")
    local_path = os.path.join("kaggle", "kaggle.json")
    if os.path.exists(default_path):
        print("Using Kaggle credentials from ~/.kaggle/kaggle.json")
        return True
    elif os.path.exists(local_path):
        print("Using Kaggle credentials from ./kaggle/kaggle.json")
        os.environ["KAGGLE_CONFIG_DIR"] = os.path.abspath("kaggle")
        return True
    else:
        print("ERROR: Kaggle API credentials not found.")
        print("Place kaggle.json in either:")
        print("  1) ~/.kaggle/kaggle.json  (chmod 600)")
        print("  2) ./kaggle/kaggle.json  (env set automatically)")
        return False


def download_and_extract_if_needed():
    if os.path.exists(AMAZON_DATA_PATH):
        print(f"Dataset already present at {AMAZON_DATA_PATH}; skipping download.")
        return

    if not ensure_kaggle_credentials():
        sys.exit(1)

    print("Downloading Amazon Product Reviews dataset from Kaggle...")
    # Download into DATA_DIR
    cmd = "kaggle datasets download -d saurav9786/amazon-product-reviews -p ./data"
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print("ERROR: Kaggle download command failed.")
        sys.exit(1)

    # Find the zip dynamically
    zip_path = None
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".zip") and "amazon-product-reviews" in fname:
            zip_path = os.path.join(DATA_DIR, fname)
            break
    if not zip_path:
        print("ERROR: Downloaded dataset zip file not found in ./data.")
        sys.exit(1)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print("Extraction complete.")


# =========================================
# Step 2: Load & Preprocess Data (build user-user interactions)
# =========================================

def load_amazon_data(path, n_rows=100000):
    # The Electronics ratings CSV has no header; columns are user, item, rating, timestamp
    df = pd.read_csv(path, names=["user", "item", "rating", "timestamp"], header=None, nrows=n_rows)
    print(f"Amazon dataset loaded with {len(df)} rows:")
    print(df.head())
    # Return only the relevant columns
    return df[["user", "item", "rating"]]


def build_user_user_weights(df):
    """Build weighted user-user edges based on co-rated items.
    Weight = number of common items rated by the pair.
    """
    # Map item -> list of users who rated it
    user_lists_by_item = df.groupby("item_id")["user_id"].apply(list)

    weight_counter = defaultdict(int)
    for users in user_lists_by_item:
        # For each pair of users who rated the same item, increment weight
        # Note: users are ints; create sorted tuple key so (u,v)==(v,u)
        for u, v in combinations(users, 2):
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            weight_counter[(a, b)] += 1

    # Build graph
    G = nx.Graph()
    # Add nodes (all users present in df)
    user_ids = df["user_id"].unique().tolist()
    G.add_nodes_from(user_ids)

    # Add weighted edges
    edges = [(u, v, {"weight": w}) for (u, v), w in weight_counter.items()]
    G.add_edges_from(edges)

    # Optional attributes: degree, weighted degree
    degree_dict = dict(G.degree())
    weighted_degree_dict = dict(G.degree(weight='weight'))
    nx.set_node_attributes(G, degree_dict, "degree")
    nx.set_node_attributes(G, weighted_degree_dict, "w_degree")

    print(f"User-User graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


# =========================================
# Step 3: Apply PageRank (Information Diffusion Proxy)
# =========================================

def apply_pagerank(G, alpha=0.85):
    print("Running PageRank (weighted)...")
    pr = nx.pagerank(G, alpha=alpha, weight='weight')
    nx.set_node_attributes(G, pr, "pagerank")
    return pr


# =========================================
# Step 4: Evaluate & Save Results
# =========================================

def save_graphml(G, path):
    nx.write_graphml(G, path)
    print(f"GraphML saved to: {path}")


def save_influencers(pr_scores, G, top_k=25):
    # Build DataFrame with degrees
    deg = dict(G.degree())
    wdeg = dict(G.degree(weight='weight'))

    df_scores = pd.DataFrame({
        'user_id': list(pr_scores.keys()),
        'pagerank': list(pr_scores.values()),
        'degree': [deg.get(u, 0) for u in pr_scores.keys()],
        'weighted_degree': [wdeg.get(u, 0.0) for u in pr_scores.keys()],
    })
    df_scores.sort_values('pagerank', ascending=False, inplace=True)

    full_path = os.path.join(DIFFUSION_DIR, 'pagerank_scores_dataset2.csv')
    df_scores.to_csv(full_path, index=False)
    print(f"All PageRank scores saved to: {full_path}")

    top_path = os.path.join(DIFFUSION_DIR, f'pagerank_top_{top_k}_influencers_dataset2.csv')
    df_scores.head(top_k).to_csv(top_path, index=False)
    print(f"Top {top_k} influencers saved to: {top_path}")

    # Print a quick preview
    print("Top influencers (preview):")
    print(df_scores.head(min(top_k, 10)).to_string(index=False))


# =========================================
# Step 5: Visualization
# =========================================

def visualize_pagerank(G, pr_scores, max_nodes=150):
    # To keep visualization readable, plot subgraph induced by top-N PageRank nodes
    top_nodes = sorted(pr_scores, key=pr_scores.get, reverse=True)[:max_nodes]
    H = G.subgraph(top_nodes).copy()

    # Normalize sizes for plotting
    pr_vals = [pr_scores[n] for n in H.nodes()]
    min_pr, max_pr = min(pr_vals), max(pr_vals)
    # Avoid zero division
    pr_norm = [((v - min_pr) / (max_pr - min_pr) if max_pr > min_pr else 1.0) for v in pr_vals]
    sizes = [300 + 3000 * v for v in pr_norm]

    pos = nx.spring_layout(H, seed=42, weight='weight')

    plt.figure(figsize=(12, 9))
    nodes = nx.draw_networkx_nodes(H, pos, node_size=sizes, node_color=pr_vals, cmap=plt.cm.Blues)
    nx.draw_networkx_edges(H, pos, alpha=0.25)
    nx.draw_networkx_labels(H, pos, font_size=7)
    plt.title("Information Diffusion via PageRank (Top Nodes)")
    cbar = plt.colorbar(nodes)
    cbar.set_label("PageRank Score")
    plt.axis('off')

    img_path = os.path.join(DIFFUSION_DIR, 'pagerank_top_nodes_dataset2.png')
    plt.savefig(img_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {img_path}")


# =========================================
# PageRank Diffusion Visualization (Snapshots)
# =========================================
def visualize_pagerank_diffusion(G, pr_scores, steps=5):
    """
    Visualize the diffusion of high PageRank scores in time-based snapshots.
    """
    sorted_nodes = sorted(pr_scores, key=pr_scores.get, reverse=True)
    num_nodes = len(sorted_nodes)
    step_size = max(1, num_nodes // steps)
    layout = nx.spring_layout(G, seed=42, weight='weight')

    for i in range(steps):
        cutoff = (i + 1) * step_size
        active_nodes = set(sorted_nodes[:cutoff])
        node_colors = []
        for n in G.nodes():
            if n in active_nodes:
                node_colors.append('blue')
            else:
                node_colors.append('lightgray')
        plt.figure(figsize=(12, 9))
        nx.draw_networkx_nodes(G, layout, node_color=node_colors, node_size=100, alpha=0.85)
        nx.draw_networkx_edges(G, layout, alpha=0.15)
        if len(G.nodes()) <= 150:
            nx.draw_networkx_labels(G, layout, font_size=7)
        plt.title(f"PageRank Diffusion Step {i+1}/{steps}")
        plt.axis('off')
        img_path = os.path.join(DIFFUSION_DIR, f'pagerank_diffusion_step_{i+1}_dataset2.png')
        plt.savefig(img_path, dpi=180, bbox_inches='tight')
        plt.close()
        print(f"Diffusion snapshot saved to: {img_path}")


def create_diffusion_gif(steps=5):
    """
    Create a GIF from the saved PageRank diffusion snapshots.
    """
    images = []
    for i in range(steps):
        img_path = os.path.join(DIFFUSION_DIR, f'pagerank_diffusion_step_{i+1}_dataset2.png')
        if os.path.exists(img_path):
            images.append(imageio.imread(img_path))
    if images:
        gif_path = os.path.join(DIFFUSION_DIR, 'pagerank_diffusion_dataset2.gif')
        imageio.mimsave(gif_path, images, duration=1)
        print(f"Diffusion GIF saved to: {gif_path}")


# =========================================
# Brute Force Community Detection (on small subgraph)
# =========================================

def compute_modularity(graph, communities):
    """Unweighted modularity for a given partition (list of sets)."""
    m = graph.number_of_edges()
    if m == 0:
        return 0.0
    deg = dict(graph.degree())
    Q = 0.0
    for community in communities:
        for u in community:
            for v in community:
                A_uv = 1 if graph.has_edge(u, v) else 0
                Q += A_uv - (deg[u] * deg[v]) / (2 * m)
    return Q / (2 * m)

def select_random_subgraph(G, size=8, max_tries=50):
    nodes = list(G.nodes())
    if len(nodes) <= size:
        H = G.copy()
        return H
    for _ in range(max_tries):
        sample = random.sample(nodes, size)
        H = G.subgraph(sample).copy()
        if H.number_of_edges() > 0:
            return H
    # Fallback: pick top-degree nodes
    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:size]
    return G.subgraph([n for n, _ in top_nodes]).copy()

def brute_force_best_partition(subgraph):
    """Search all 2-way splits (by choosing one group size 2..n//2)."""
    nodes = list(subgraph.nodes())
    best_partition = None
    best_mod = float('-inf')
    from itertools import combinations
    for r in range(2, len(nodes)//2 + 1):
        for group in combinations(nodes, r):
            c1 = set(group)
            c2 = set(nodes) - c1
            mod = compute_modularity(subgraph, [c1, c2])
            if mod > best_mod:
                best_mod = mod
                best_partition = [c1, c2]
    return best_partition, best_mod

def visualize_communities(subgraph, partition, modularity_value, out_path):
    colors = []
    for n in subgraph.nodes():
        colors.append('lightblue' if n in partition[0] else 'lightgreen')
    pos = nx.spring_layout(subgraph, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw_networkx(subgraph, pos, with_labels=True, node_color=colors, node_size=800, font_size=8)
    plt.title(f"Brute Force Communities (Q={modularity_value:.4f})")
    plt.axis('off')
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()

# =========================================
# Recommender System Functions
# =========================================

def plot_histogram(data, xlabel, ylabel, title, save_path):
    plt.figure(figsize=(8,6))
    plt.hist(data, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved histogram to: {save_path}")

def create_user_item_matrix(df):
    # Pivot ratings to create user-item matrix
    user_item = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    return user_item

def cosine_sim_user(user_item_matrix):
    # Compute cosine similarity between users
    sim_matrix = cosine_similarity(user_item_matrix)
    np.fill_diagonal(sim_matrix, 0)  # zero out self-similarity
    return pd.DataFrame(sim_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

def recommend_for_user(user_id, user_item_matrix, user_sim_matrix, top_n=10):
    if user_id not in user_item_matrix.index:
        return []

    # Get similarity scores for the user with all others
    sim_scores = user_sim_matrix.loc[user_id]

    # Weighted sum of ratings from similar users
    ratings = user_item_matrix.loc[user_item_matrix.index != user_id]
    sim_scores = sim_scores.loc[ratings.index]

    weighted_ratings = ratings.mul(sim_scores, axis=0)
    recommendation_scores = weighted_ratings.sum(axis=0) / (sim_scores.sum() + 1e-9)

    # Filter out items already rated by user
    user_rated_items = user_item_matrix.loc[user_id]
    unrated_items = user_rated_items[user_rated_items == 0].index

    # Recommend top N items
    recs = recommendation_scores.loc[unrated_items].sort_values(ascending=False).head(top_n)
    return list(recs.index)

def precision_recall_f1_at_k(test_df, train_df, user_item_matrix, user_sim_matrix, k=10):
    users = test_df['user_id'].unique()
    precisions = []
    recalls = []
    f1s = []

    # Build test user-item dict
    test_user_items = test_df.groupby('user_id')['item_id'].apply(set).to_dict()
    # Build train user-item dict
    train_user_items = train_df.groupby('user_id')['item_id'].apply(set).to_dict()

    for user in users:
        true_items = test_user_items.get(user, set())
        if not true_items:
            continue
        recommended_items = recommend_for_user(user, user_item_matrix, user_sim_matrix, top_n=k)
        recommended_set = set(recommended_items)

        tp = len(true_items & recommended_set)
        fp = len(recommended_set - true_items)
        fn = len(true_items - recommended_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_recall = np.mean(recalls) if recalls else 0.0
    avg_f1 = np.mean(f1s) if f1s else 0.0
    return avg_precision, avg_recall, avg_f1


# =========================================
# Main Execution
# =========================================
if __name__ == "__main__":
    download_and_extract_if_needed()

    # Load Amazon ratings
    if not os.path.exists(AMAZON_DATA_PATH):
        print(f"ERROR: Expected ratings file not found at {AMAZON_DATA_PATH}")
        sys.exit(1)
    ratings_raw = load_amazon_data(AMAZON_DATA_PATH, n_rows=100000)

    # Map string username/asins to integer IDs for internal use
    user2id = {u: i for i, u in enumerate(ratings_raw["user"].unique())}
    item2id = {a: i for i, a in enumerate(ratings_raw["item"].unique())}
    ratings = ratings_raw.copy()
    ratings["user_id"] = ratings["user"].map(user2id)
    ratings["item_id"] = ratings["item"].map(item2id)
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors='coerce')
    ratings = ratings.dropna(subset=["user_id", "item_id", "rating"])
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["item_id"] = ratings["item_id"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)
    # Sample a smaller subset for graph operations
    ratings = ratings.sample(n=50000, random_state=42)
    print(f"Sampled down to {len(ratings)} rows for graph and recommendation tasks.")

    # Build user-user weighted graph
    G = build_user_user_weights(ratings)

    # Apply PageRank
    pr_scores = apply_pagerank(G, alpha=0.85)

    # Save results
    save_graphml(G, os.path.join(DIFFUSION_DIR, 'pagerank_user_user_graph_dataset2.graphml'))
    save_influencers(pr_scores, G, top_k=25)

    # Visualize
    visualize_pagerank(G, pr_scores, max_nodes=150)
    visualize_pagerank_diffusion(G, pr_scores, steps=5)
    create_diffusion_gif(steps=5)

    print("\nDone with PageRank pipeline.\n")

    # =========================================
    # Community Detection Pipeline (Brute Force on small subgraph)
    # =========================================
    print("=== Community Detection (Brute Force) Pipeline ===")

    SUBGRAPH_SIZE = 8
    subgraph = select_random_subgraph(G, size=SUBGRAPH_SIZE)
    print(f"Selected subgraph with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges.")

    best_partition, best_modularity = brute_force_best_partition(subgraph)
    sizes = [len(c) for c in best_partition] if best_partition else []

    # Save community results
    community_txt = os.path.join(COMMUNITY_DIR, 'community_partition_dataset2.txt')
    with open(community_txt, 'w') as f:
        f.write(f"Best modularity: {best_modularity}\n")
        if best_partition:
            for i, c in enumerate(best_partition, 1):
                f.write(f"Community {i} (size {len(c)}): {sorted(list(c))}\n")
        f.write(f"Number of communities: {len(best_partition) if best_partition else 0}\n")
        f.write(f"Size distribution: {sizes}\n")
    print(f"Community partition saved to: {community_txt}")

    # Save subgraph GraphML
    community_graphml = os.path.join(COMMUNITY_DIR, 'community_subgraph_dataset2.graphml')
    nx.write_graphml(subgraph, community_graphml)
    print(f"Community subgraph saved to: {community_graphml}")

    # Visualization
    community_png = os.path.join(COMMUNITY_DIR, 'community_subgraph_dataset2.png')
    visualize_communities(subgraph, best_partition, best_modularity, community_png)
    print(f"Community visualization saved to: {community_png}")

    print("\nDone with Community Detection pipeline.\n")

    # =========================================
    # Recommender System Pipeline
    # =========================================
    print("=== Recommender System Pipeline ===")

    # a. Data Cleaning & Type Conversion (reuse ratings)
    # Already handled above

    # b. EDA: Distribution of users, items, ratings
    num_users = ratings['user_id'].nunique()
    num_items = ratings['item_id'].nunique()
    num_ratings = len(ratings)

    print(f"Number of users: {num_users}")
    print(f"Number of items: {num_items}")
    print(f"Number of ratings: {num_ratings}")

    # Histograms
    plot_histogram(ratings['user_id'], 'User ID', 'Count', 'Distribution of Users', os.path.join(RECOMMENDER_DIR, 'hist_users_dataset2.png'))
    plot_histogram(ratings['item_id'], 'Item ID', 'Count', 'Distribution of Items', os.path.join(RECOMMENDER_DIR, 'hist_items_dataset2.png'))
    plot_histogram(ratings['rating'], 'Rating', 'Count', 'Distribution of Ratings', os.path.join(RECOMMENDER_DIR, 'hist_ratings_dataset2.png'))

    # c. Split data: 80/20 train/test
    train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)

    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")

    # d. Apply User-based Collaborative Filtering (cosine similarity)
    user_item_train = create_user_item_matrix(train_df)
    user_sim_matrix = cosine_sim_user(user_item_train)

    print("User-based collaborative filtering model created.")

    # e. Make predictions: top-N recommendations for sample users
    sample_users = user_item_train.index[:5]  # first 5 users in train set
    recommendations = []
    for user in sample_users:
        rec_items = recommend_for_user(user, user_item_train, user_sim_matrix, top_n=10)
        for item in rec_items:
            recommendations.append({'user_id': user, 'recommended_item': item})

    rec_df = pd.DataFrame(recommendations)
    rec_csv_path = os.path.join(RECOMMENDER_DIR, 'recommendations_dataset2.csv')
    rec_df.to_csv(rec_csv_path, index=False)
    print(f"Recommendations saved to: {rec_csv_path}")

    # f. Evaluate: Precision@K, Recall@K, F1-score on test data
    precision, recall, f1 = precision_recall_f1_at_k(test_df, train_df, user_item_train, user_sim_matrix, k=10)

    eval_path = os.path.join(RECOMMENDER_DIR, 'evaluation_dataset2.txt')
    with open(eval_path, 'w') as f:
        f.write(f"Precision@10: {precision:.4f}\n")
        f.write(f"Recall@10: {recall:.4f}\n")
        f.write(f"F1-score@10: {f1:.4f}\n")
    print(f"Evaluation metrics saved to: {eval_path}")

    print("\nRecommender System Pipeline complete.\n")
