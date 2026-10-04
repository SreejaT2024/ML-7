import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ============================================================
# A1. Own functions: summation, activations, comparator (error)
# ============================================================
def summation(inputs, weights, bias):
    """y = w0 + sum(wi * xi)"""
    return bias + sum(x * w for x, w in zip(inputs, weights))

def step(y):            return 1.0 if y > 0 else 0.0
def bipolar_step(y):    return 1.0 if y > 0 else (0.0 if y == 0 else -1.0)
def sigmoid(y):         return 1.0 / (1.0 + np.exp(-y))
def tanh(y):            return (np.exp(y) - np.exp(-y)) / (np.exp(y) + np.exp(-y))
def relu(y):            return max(0.0, y)
def leaky_relu(y, a=0.01): return y if y > 0 else a * y

def error_unit(target, output):
    """Comparator: error for one sample"""
    return target - output

# ============================================================
# Data
# ============================================================
X_in = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
T_AND = np.array([0, 0, 0, 1], dtype=float)
T_XOR = np.array([0, 1, 1, 0], dtype=float)

W_INIT = (10.0, 0.2, -0.75)   # W0, W1, W2
LR = 0.05
MAX_EPOCHS = 1000
TOL = 0.002

# ============================================================
# A2. Perceptron (own code, delta rule  w += lr*(t-o)*x)
# ============================================================
def train_perceptron(X, T, act, lr=LR, w_init=W_INIT, max_epochs=MAX_EPOCHS, tol=TOL):
    w0, w1, w2 = w_init
    errors = []
    for epoch in range(1, max_epochs + 1):
        sse = 0.0
        for x, t in zip(X, T):
            y = summation(x, (w1, w2), w0)
            o = act(y)
            e = error_unit(t, o)
            w0 += lr * e * 1
            w1 += lr * e * x[0]
            w2 += lr * e * x[1]
        # sum-square-error after the epoch over all samples
        for x, t in zip(X, T):
            o = act(summation(x, (w1, w2), w0))
            sse += (t - o) ** 2
        errors.append(sse)
        if sse <= tol:
            return (w0, w1, w2), errors, epoch, True
    return (w0, w1, w2), errors, max_epochs, False

def plot_errors(errors, title):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(errors) + 1), errors)
    plt.xlabel("Epochs"); plt.ylabel("Sum-square error"); plt.title(title)
    plt.grid(True); plt.tight_layout()
    plt.savefig(title.replace(" ", "_").replace("/", "_") + ".png"); plt.close()

def a2():
    print("\n=== A2: AND gate, Step activation ===")
    w, errs, ep, conv = train_perceptron(X_in, T_AND, step)
    print("Weights:", w, "| epochs:", ep, "| converged:", conv)
    plot_errors(errs, "A2 AND Step")

# ============================================================
# A3. Other activations on AND
# ============================================================
def a3(T=T_AND, name="AND", tag="A3"):
    print(f"\n=== {tag}: {name} gate, different activations ===")
    results = {}
    for aname, act, bipolar in [("Step", step, False), ("Bipolar Step", bipolar_step, True),
                                ("Sigmoid", sigmoid, False), ("ReLU", relu, False)]:
        Tt = 2 * T - 1 if bipolar else T       # bipolar step outputs -1/1, so targets mapped to -1/1
        w, errs, ep, conv = train_perceptron(X_in, Tt, act)
        results[aname] = ep
        print(f"{aname:13s} epochs = {ep:4d} converged = {conv}")
        plot_errors(errs, f"{tag} {name} {aname}")
    return results

# ============================================================
# A4. Varying learning rate (AND, step)
# ============================================================
def a4(T=T_AND, name="AND", tag="A4"):
    print(f"\n=== {tag}: learning-rate variation ({name}, Step) ===")
    lrs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    iters = []
    for lr in lrs:
        _, _, ep, conv = train_perceptron(X_in, T, step, lr=lr)
        iters.append(ep)
        print(f"lr={lr}: epochs={ep}, converged={conv}")
    plt.figure(figsize=(6, 4))
    plt.plot(lrs, iters, marker="o")
    plt.xlabel("Learning rate"); plt.ylabel("Epochs to converge")
    plt.title(f"{tag} {name} lr vs epochs"); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{tag}_{name}_lr_vs_epochs.png"); plt.close()

# ============================================================
# A5. Repeat A1-A3 (and A4) for XOR
# ============================================================
def a5():
    print("\n=== A5: XOR gate (single perceptron cannot solve XOR: not linearly separable) ===")
    w, errs, ep, conv = train_perceptron(X_in, T_XOR, step)
    print("Step ->", w, "epochs:", ep, "converged:", conv)
    plot_errors(errs, "A5 XOR Step")
    a3(T_XOR, "XOR", "A5")
    a4(T_XOR, "XOR", "A5")

# ============================================================
# A6. Customer data, sigmoid perceptron
# ============================================================
CUST = np.array([
    [20, 6, 2, 386], [16, 3, 6, 289], [27, 6, 2, 393], [19, 1, 2, 110], [24, 4, 2, 280],
    [22, 1, 5, 167], [15, 4, 2, 271], [18, 4, 2, 274], [21, 1, 4, 148], [16, 2, 4, 198]], dtype=float)
Y_CUST = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0], dtype=float)   # Yes=1, No=0

def a6(lr=0.1, epochs=5000, seed=1):
    print("\n=== A6: customer classification (sigmoid perceptron) ===")
    rng = np.random.default_rng(seed)
    Xn = (CUST - CUST.mean(0)) / CUST.std(0)          # normalise features
    w = rng.uniform(-0.5, 0.5, Xn.shape[1]); b = rng.uniform(-0.5, 0.5)
    errs = []
    for ep in range(epochs):
        for x, t in zip(Xn, Y_CUST):
            o = sigmoid(summation(x, w, b))
            e = t - o
            w = w + lr * e * x
            b = b + lr * e
        o_all = np.array([sigmoid(summation(x, w, b)) for x in Xn])
        sse = np.sum((Y_CUST - o_all) ** 2)
        errs.append(sse)
        if sse <= TOL: break
    pred = (o_all >= 0.5).astype(int)
    print("epochs:", ep + 1, "SSE:", round(sse, 4))
    print("weights:", w, "bias:", b)
    print("predictions:", pred, "\nactual:     ", Y_CUST.astype(int))
    print("accuracy:", (pred == Y_CUST).mean())
    plot_errors(errs, "A6 customer sigmoid")
    return Xn, w, b, pred

# ============================================================
# A7. Matrix pseudo-inverse comparison
# ============================================================
def a7(Xn, pred_perceptron):
    print("\n=== A7: pseudo-inverse solution ===")
    A = np.hstack([np.ones((Xn.shape[0], 1)), Xn])     # bias column
    w = np.linalg.pinv(A) @ Y_CUST
    scores = A @ w
    pred = (scores >= 0.5).astype(int)
    print("weights (bias first):", w)
    print("pinv predictions:      ", pred)
    print("perceptron predictions:", pred_perceptron)
    print("pinv accuracy:", (pred == Y_CUST).mean(),
          "| perceptron accuracy:", (pred_perceptron == Y_CUST).mean())

# ============================================================
# A8 / A9. 2-2-1 network with back-propagation (Mitchell)
# ============================================================
def train_bp(X, T, lr=LR, max_epochs=MAX_EPOCHS, tol=TOL, seed=0, n_hidden=2):
    rng = np.random.default_rng(seed)
    V = rng.uniform(-0.5, 0.5, (X.shape[1] + 1, n_hidden))   # input->hidden (row 0 = bias)
    W = rng.uniform(-0.5, 0.5, (n_hidden + 1, 1))             # hidden->output (row 0 = bias)
    errs = []
    for ep in range(1, max_epochs + 1):
        for x, t in zip(X, T):
            xi = np.concatenate([[1], x])
            h = sigmoid_v(xi @ V)
            hi = np.concatenate([[1], h])
            o = sigmoid_v(hi @ W)
            d_o = o * (1 - o) * (t - o)                         # T4.3
            d_h = h * (1 - h) * (W[1:, 0] * d_o[0])             # T4.4
            W += lr * np.outer(hi, d_o)                         # T4.5
            V += lr * np.outer(xi, d_h)
        sse = 0.0
        for x, t in zip(X, T):
            h = sigmoid_v(np.concatenate([[1], x]) @ V)
            o = sigmoid_v(np.concatenate([[1], h]) @ W)
            sse += (t - o[0]) ** 2
        errs.append(sse)
        if sse <= tol: return errs, ep, True, (V, W)
    return errs, max_epochs, False, (V, W)

def sigmoid_v(z): return 1 / (1 + np.exp(-z))

def a8_a9():
    for name, T, tag in [("AND", T_AND, "A8"), ("XOR", T_XOR, "A9")]:
        errs, ep, conv, (V, W) = train_bp(X_in, T)
        print(f"\n=== {tag}: backprop {name} === epochs={ep} converged={conv} final SSE={errs[-1]:.4f}")
        plot_errors(errs, f"{tag} backprop {name}")
        # XOR typically needs more than 1000 epochs at lr=0.05; show with more epochs for reference
        if not conv:
            errs2, ep2, conv2, _ = train_bp(X_in, T, max_epochs=200000)
            print(f"   (with up to 200000 epochs: epochs={ep2}, converged={conv2})")

# ============================================================
# A10. Two output nodes ([1 0] for 0, [0 1] for 1)
# ============================================================
def a10(T=T_AND, name="AND"):
    print(f"\n=== A10: {name}, 2 output nodes ===")
    T2 = np.array([[1, 0] if t == 0 else [0, 1] for t in T], dtype=float)
    for aname, act in [("Step", step), ("Sigmoid", sigmoid)]:
        W = np.array([[10.0, 10.0], [0.2, 0.2], [-0.75, -0.75]])    # rows: w0,w1,w2; cols: O1,O2
        errs = []
        for ep in range(1, MAX_EPOCHS + 1):
            for x, t in zip(X_in, T2):
                for k in range(2):
                    o = act(summation(x, W[1:, k], W[0, k]))
                    e = t[k] - o
                    W[0, k] += LR * e; W[1, k] += LR * e * x[0]; W[2, k] += LR * e * x[1]
            sse = sum((T2[i, k] - act(summation(X_in[i], W[1:, k], W[0, k]))) ** 2
                      for i in range(4) for k in range(2))
            errs.append(sse)
            if sse <= TOL: break
        print(f"{aname}: epochs={ep}, SSE={sse:.4f}, converged={sse <= TOL}")
        plot_errors(errs, f"A10 {name} 2out {aname}")

# ============================================================
# A11. sklearn MLPClassifier for AND / XOR
# ============================================================
def a11():
    from sklearn.neural_network import MLPClassifier
    print("\n=== A11: MLPClassifier ===")
    for name, T in [("AND", T_AND), ("XOR", T_XOR)]:
        clf = MLPClassifier(hidden_layer_sizes=(4,), activation="logistic", solver="lbfgs",
                            max_iter=2000, random_state=1)
        clf.fit(X_in, T.astype(int))
        print(name, "predictions:", clf.predict(X_in), "| accuracy:", clf.score(X_in, T.astype(int)))

# ============================================================
# A12. MLPClassifier on YOUR project dataset (edit the path / target column)
# ============================================================
def a12(csv_path="your_dataset.csv", target="target"):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    df = pd.read_csv(csv_path)
    X = pd.get_dummies(df.drop(columns=[target]))
    y = df[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler().fit(Xtr)
    clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)
    clf.fit(sc.transform(Xtr), ytr)
    print("Train acc:", clf.score(sc.transform(Xtr), ytr), "| Test acc:", clf.score(sc.transform(Xte), yte))

# ============================================================
if __name__ == "__main__":
    matplotlib.use("Agg")      # remove this line if you want plots to pop up
    a2()
    a3()
    a4()
    a5()
    Xn, w, b, pred = a6()
    a7(Xn, pred)
    a8_a9()
    a10(T_AND, "AND")
    a10(T_XOR, "XOR")
    a11()
    # a12("your_dataset.csv", "target")   # uncomment after setting your dataset
