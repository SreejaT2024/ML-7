"""
Decision Tree assignment (A1 - A8) on the IMDb Top-250 movies dataset.

Problem set-up
--------------
Target  : averageRating  (continuous -> regression style outcome)
          A1/A2 : the target is binned into 4 equal-width bins (as the question says)
          A3-A8 : those 4 bins are the class labels for the classifier
Features: runtimeMinutes, numVotes, metascore, startYear, log(grossWorldwide),
          contentRating, mainGenre
"""
import ast
import math
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

CSV_PATH = "ToP_250_movies_on_imdb_in_2026.csv"  # change if your file is elsewhere


# ============================================================
# Binning  (A1 equal-width  +  A4 equal-width / frequency)
# ============================================================
def equal_width_edges(values, n_bins=4):
    """Bin edges for equal-width binning."""
    v = np.asarray(values, dtype=float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    return np.linspace(lo, hi, n_bins + 1)


def equal_frequency_edges(values, n_bins=4):
    """Bin edges for equal-frequency (quantile) binning."""
    v = np.asarray(values, dtype=float)
    edges = np.quantile(v, np.linspace(0, 1, n_bins + 1))
    return np.unique(edges)  # drop duplicate edges if many ties


def apply_edges(values, edges):
    """Assign every value to a bin index 0..len(edges)-2 using given edges."""
    v = np.asarray(values, dtype=float)
    # use the inner edges only, so out-of-range values go to first / last bin
    return np.digitize(v, edges[1:-1], right=False)


def binning(values, n_bins=4, bin_type="width", return_edges=False):
    """
    Own binning function (A4).
    Function 'overloading' in Python is done with default parameters:
        binning(x)                          -> 4 equal-width bins
        binning(x, 6)                       -> 6 equal-width bins
        binning(x, 5, "frequency")          -> 5 equal-frequency bins
    """
    if bin_type in ("width", "equal_width"):
        edges = equal_width_edges(values, n_bins)
    elif bin_type in ("frequency", "equal_frequency"):
        edges = equal_frequency_edges(values, n_bins)
    else:
        raise ValueError("bin_type must be 'width' or 'frequency'")
    labels = apply_edges(values, edges)
    return (labels, edges) if return_edges else labels


# ============================================================
# A1  Entropy      A2  Gini
# ============================================================
def entropy(labels):
    """H = - sum p_i log2 p_i   (labels must be categorical)."""
    n = len(labels)
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in Counter(labels).values())


def entropy_continuous(values, n_bins=4):
    """A1: entropy of a continuous outcome using equal-width binning (4 bins)."""
    return entropy(binning(values, n_bins, "width"))


def gini_index(labels):
    """Gini = 1 - sum p_j^2."""
    n = len(labels)
    if n == 0:
        return 0.0
    return 1.0 - sum((c / n) ** 2 for c in Counter(labels).values())


# ============================================================
# A3  Information gain + root-node selection
# ============================================================
def information_gain(feature, labels):
    """IG = H(labels) - sum_v (|S_v|/|S|) H(S_v) for a categorical feature."""
    feature, labels = np.asarray(feature), np.asarray(labels)
    n = len(labels)
    remainder = 0.0
    for v in np.unique(feature):
        mask = feature == v
        remainder += mask.sum() / n * entropy(labels[mask])
    return entropy(labels) - remainder


def find_root_feature(X, y, n_bins=4, bin_type="width"):
    """
    A3 + A4: return the feature with the highest information gain.
    Continuous columns are converted to categorical with binning().
    """
    gains = {}
    for col in X.columns:
        col_vals = X[col]
        if pd.api.types.is_numeric_dtype(col_vals):
            col_vals = binning(col_vals.values, n_bins, bin_type)
        gains[col] = information_gain(col_vals, y)
    best = max(gains, key=gains.get)
    return best, gains


# ============================================================
# A5  Own Decision Tree (ID3, multiway splits on categorical features)
# ============================================================
class Node:
    def __init__(self, prediction, n_samples, impurity, class_counts):
        self.prediction = prediction      # majority class
        self.n_samples = n_samples
        self.impurity = impurity          # entropy at this node
        self.class_counts = class_counts
        self.feature = None               # split feature (None => leaf)
        self.children = {}                # feature value -> Node
        self.gain = 0.0


class MyDecisionTree:
    def __init__(self, max_depth=4, min_samples_split=5, min_gain=1e-4,
                 n_bins=4, bin_type="width"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.n_bins = n_bins
        self.bin_type = bin_type
        self.root = None
        self.bin_edges_ = {}

    # ---- preprocessing: continuous -> categorical (edges learned on train)
    def _fit_transform(self, X):
        X = X.copy()
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                labels, edges = binning(X[col].values, self.n_bins,
                                        self.bin_type, return_edges=True)
                self.bin_edges_[col] = edges
                X[col] = labels
        return X

    def _transform(self, X):
        X = X.copy()
        for col, edges in self.bin_edges_.items():
            X[col] = apply_edges(X[col].values, edges)
        return X

    # ---- tree building
    def fit(self, X, y):
        X = self._fit_transform(pd.DataFrame(X))
        y = np.asarray(y)
        self.classes_ = sorted(set(y))
        self.root = self._build(X.reset_index(drop=True), y, depth=0,
                                available=list(X.columns))
        return self

    def _make_node(self, y):
        counts = Counter(y)
        return Node(counts.most_common(1)[0][0], len(y), entropy(y), dict(counts))

    def _build(self, X, y, depth, available):
        node = self._make_node(y)
        if (depth >= self.max_depth or len(y) < self.min_samples_split
                or node.impurity == 0 or not available):
            return node
        gains = {c: information_gain(X[c].values, y) for c in available}
        best = max(gains, key=gains.get)
        if gains[best] < self.min_gain:
            return node
        node.feature, node.gain = best, gains[best]
        rest = [c for c in available if c != best]
        for v in np.unique(X[best].values):
            mask = (X[best] == v).values
            node.children[v] = self._build(X[mask].reset_index(drop=True),
                                           y[mask], depth + 1, rest)
        return node

    # ---- prediction
    def _predict_row(self, row):
        node = self.root
        while node.feature is not None:
            v = row[node.feature]
            if v not in node.children:      # unseen value -> stop here
                break
            node = node.children[v]
        return node.prediction

    def predict(self, X):
        X = self._transform(pd.DataFrame(X))
        return np.array([self._predict_row(r) for _, r in X.iterrows()])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


# ============================================================
# A6  Visualise the tree (own drawing with matplotlib)
# ============================================================
def draw_tree(tree, class_names=None, feature_names_edges=True, title="Own Decision Tree",
              save_as="my_tree.png"):
    root = tree.root
    # assign x positions: leaves get consecutive slots, parents are centred
    counter = [0]
    pos = {}

    def layout(node, depth):
        if node.feature is None:
            pos[id(node)] = (counter[0], -depth)
            counter[0] += 1
        else:
            xs = [layout(ch, depth + 1) for ch in node.children.values()]
            pos[id(node)] = (np.mean(xs), -depth)
        return pos[id(node)][0]

    layout(root, 0)
    fig, ax = plt.subplots(figsize=(max(12, counter[0] * 1.6), 7))
    ax.axis("off")
    ax.set_title(title, fontsize=14)

    def bin_text(feat, v):
        e = tree.bin_edges_.get(feat)
        if e is None:
            return str(v)
        lo = e[v] if v < len(e) - 1 else e[-2]
        hi = e[v + 1] if v + 1 < len(e) else e[-1]
        return f"{lo:.4g}–{hi:.4g}"

    def draw(node):
        x, y = pos[id(node)]
        name = (class_names[node.prediction] if class_names else str(node.prediction))
        if node.feature is None:
            label = f"class: {name}\nn={node.n_samples}\nH={node.impurity:.2f}"
            fc = "#d9f2d9"
        else:
            label = (f"{node.feature}\nIG={node.gain:.3f}\nn={node.n_samples}\n"
                     f"H={node.impurity:.2f}\nmajor: {name}")
            fc = "#dbe9ff"
        ax.text(x, y, label, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round", fc=fc, ec="k"))
        for v, ch in node.children.items():
            cx, cy = pos[id(ch)]
            ax.plot([x, cx], [y - 0.12, cy + 0.12], "k-", lw=0.8, zorder=0)
            ax.text((x + cx) / 2, (y + cy) / 2, bin_text(node.feature, v),
                    fontsize=7, ha="center", color="darkred",
                    bbox=dict(fc="white", ec="none", pad=0.5))
            draw(ch)

    draw(root)
    ax.set_xlim(-0.7, counter[0] - 0.3)
    ax.set_ylim(-(tree.max_depth + 0.7), 0.7)
    plt.tight_layout()
    plt.savefig(save_as, dpi=150)
    plt.show()


# ============================================================
# A7  Decision boundary for 2 features
# ============================================================
def plot_decision_boundary(model, X2, y, feature_names, class_names, title, save_as,
                           is_own=False):
    X2 = np.asarray(X2, dtype=float)
    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if is_own:
        grid = pd.DataFrame(grid, columns=feature_names)
    Z = np.asarray(model.predict(grid)).reshape(xx.shape)

    cmap_bg = ListedColormap(["#ffd6d6", "#d6e4ff", "#d6ffd9", "#fff3c4"][:len(class_names)])
    cmap_pt = ListedColormap(["#d62728", "#1f77b4", "#2ca02c", "#e6a800"][:len(class_names)])
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.6, cmap=cmap_bg,
                 levels=np.arange(-0.5, len(class_names) + 0.5, 1))
    sc = plt.scatter(X2[:, 0], X2[:, 1], c=y, cmap=cmap_pt, edgecolor="k", s=25,
                     vmin=0, vmax=len(class_names) - 1)
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=cmap_pt(i), mec="k", label=n)
               for i, n in enumerate(class_names)]
    plt.legend(handles=handles, title="Rating class")
    plt.xlabel(feature_names[0]); plt.ylabel(feature_names[1]); plt.title(title)
    plt.tight_layout(); plt.savefig(save_as, dpi=150); plt.show()


# ============================================================
# Data loading / preparation
# ============================================================
def load_data(path=CSV_PATH):
    df = pd.read_csv(path)
    df["mainGenre"] = df["genres"].apply(lambda s: ast.literal_eval(s)[0] if s else "Unknown")
    df["logGross"] = np.log10(df["grossWorldwide"])
    df["contentRating"] = df["contentRating"].fillna("Unknown")
    for c in ["metascore", "logGross"]:
        df[c] = df[c].fillna(df[c].median())
    return df


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    df = load_data()
    rating = df["averageRating"].values

    # ---------------- A1
    print("=== A1: Entropy of the outcome (averageRating, 4 equal-width bins) ===")
    y_bins, edges = binning(rating, 4, "width", return_edges=True)
    print("Bin edges :", np.round(edges, 3))
    print("Bin counts:", dict(sorted(Counter(y_bins).items())))
    print("Entropy   :", round(entropy_continuous(rating), 4))

    # ---------------- A2
    print("\n=== A2: Gini index ===")
    print("Gini      :", round(gini_index(y_bins), 4))

    # ---------------- A3 / A4
    feature_cols = ["runtimeMinutes", "numVotes", "metascore", "startYear",
                    "logGross", "contentRating", "mainGenre"]
    X = df[feature_cols]
    print("\n=== A3/A4: Root node by information gain ===")
    for bt in ["width", "frequency"]:
        root, gains = find_root_feature(X, y_bins, n_bins=4, bin_type=bt)
        print(f"[{bt} binning] root = {root}")
        for k, v in sorted(gains.items(), key=lambda kv: -kv[1]):
            print(f"    {k:15s} IG = {v:.4f}")

    # ---------------- A5
    class_names = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(4)]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_bins, test_size=0.25,
                                              random_state=42, stratify=y_bins)
    print("\n=== A5: Own Decision Tree ===")
    my_tree = MyDecisionTree(max_depth=3, min_samples_split=5,
                             n_bins=4, bin_type="frequency").fit(X_tr, y_tr)
    print("Train accuracy:", round(my_tree.score(X_tr, y_tr), 3))
    print("Test  accuracy:", round(my_tree.score(X_te, y_te), 3))

    # ---------------- A6
    print("\n=== A6: Tree visualisation ===")
    draw_tree(my_tree, class_names=class_names,
              title="Own Decision Tree (target = rating class)", save_as="my_tree.png")

    # sklearn reference tree (numeric features only)
    num_cols = ["runtimeMinutes", "numVotes", "metascore", "startYear", "logGross"]
    sk = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
    sk.fit(X_tr[num_cols], y_tr)
    plt.figure(figsize=(18, 8))
    plot_tree(sk, feature_names=num_cols, class_names=class_names, filled=True, rounded=True)
    plt.savefig("sklearn_tree.png", dpi=150); plt.show()

    # ---------------- A7
    print("\n=== A7: Decision boundary using 2 features ===")
    two = ["metascore", "runtimeMinutes"]
    X2_tr, X2_te, y2_tr, y2_te = train_test_split(df[two], y_bins, test_size=0.25,
                                                  random_state=42, stratify=y_bins)
    my2 = MyDecisionTree(max_depth=4, n_bins=6, bin_type="frequency").fit(X2_tr, y2_tr)
    print("Own tree  test acc:", round(my2.score(X2_te, y2_te), 3))
    plot_decision_boundary(my2, df[two].values, y_bins, two, class_names,
                           "Decision boundary – own tree", "boundary_own.png", is_own=True)

    sk2 = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42).fit(X2_tr, y2_tr)
    print("sklearn   test acc:", round(sk2.score(X2_te, y2_te), 3))
    plot_decision_boundary(sk2, df[two].values, y_bins, two, class_names,
                           "Decision boundary – sklearn tree", "boundary_sklearn.png")

    # ---------------- A8
    print("\n=== A8: Hyper-parameter tuning ===")
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [2, 3, 4, 5, 6, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
    }
    Xn_tr, Xn_te = X_tr[num_cols], X_te[num_cols]
    grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid,
                        cv=5, scoring="accuracy", n_jobs=-1).fit(Xn_tr, y_tr)
    print("GridSearchCV   best params:", grid.best_params_)
    print("               CV acc     :", round(grid.best_score_, 3),
          "| test acc:", round(grid.score(Xn_te, y_te), 3))

    rand = RandomizedSearchCV(DecisionTreeClassifier(random_state=42),
                              {**param_grid, "max_depth": list(range(2, 12)) + [None],
                               "min_samples_split": list(range(2, 30))},
                              n_iter=40, cv=5, scoring="accuracy",
                              random_state=42, n_jobs=-1).fit(Xn_tr, y_tr)
    print("RandomizedSearchCV best params:", rand.best_params_)
    print("               CV acc     :", round(rand.best_score_, 3),
          "| test acc:", round(rand.score(Xn_te, y_te), 3))
