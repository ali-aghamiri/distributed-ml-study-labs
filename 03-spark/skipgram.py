"""Small from-scratch skip-gram with negative sampling and Spark gradients."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import re
import sys

import numpy as np


def words(line):
    if line.lstrip().startswith("#"):
        return []
    return re.findall(r"[^\W\d_]+", line, flags=re.UNICODE)


def pairs(tokens, lookup, window):
    ids = [lookup[t] for t in tokens if t in lookup]
    for i, center in enumerate(ids):
        for j in range(max(0, i-window), min(len(ids), i+window+1)):
            if i != j:
                yield center, ids[j]


def partition_gradients(iterator, inputs, outputs, seed, negatives):
    rng = np.random.default_rng(seed)
    gradients = defaultdict(lambda: np.zeros(inputs.shape[1], dtype=np.float64))
    vocab_size = len(inputs)
    for center, context in iterator:
        for target, label in [(context, 1)] + [
                (int(rng.integers(vocab_size)), 0) for _ in range(negatives)]:
            score = np.clip(inputs[center] @ outputs[target], -30, 30)
            error = 1 / (1 + np.exp(-score)) - label
            gradients[(0, center)] += error * outputs[target]
            gradients[(1, target)] += error * inputs[center]
    return iter(gradients.items())


def load_local(path, window, max_pairs):
    with open(path, encoding="utf-8-sig") as stream:
        lines = [words(line) for line in stream]
    counts = Counter(word for line in lines for word in line)
    vocab = sorted(counts, key=lambda word: (-counts[word], word))
    lookup = {word: i for i, word in enumerate(vocab)}
    examples = [pair for line in lines for pair in pairs(line, lookup, window)]
    if len(examples) > max_pairs:
        selected = np.random.default_rng(123).choice(
            len(examples), size=max_pairs, replace=False)
        examples = [examples[i] for i in selected]
    return vocab, examples


def train_local(vocab, examples, dim, epochs, negatives, lr):
    rng = np.random.default_rng(123)
    inputs = rng.normal(0, 0.1, (len(vocab), dim))
    outputs = rng.normal(0, 0.1, inputs.shape)
    for epoch in range(epochs):
        order = rng.permutation(len(examples))
        for start in range(0, len(examples), 256):
            batch = [examples[i] for i in order[start:start+256]]
            grads = partition_gradients(iter(batch), inputs, outputs,
                                        123 + epoch * 100000 + start, negatives)
            for (which, index), gradient in grads:
                (inputs if which == 0 else outputs)[index] -= lr * gradient / len(batch)
    return inputs


def train_spark(path, window, max_pairs, dim, epochs, negatives, lr):
    from pyspark import SparkContext
    sc = SparkContext.getOrCreate()
    lines = sc.textFile(path).map(words).cache()
    counts = lines.flatMap(lambda line: [(word, 1) for word in line]).reduceByKey(
        lambda a, b: a + b).collect()
    vocab = [word for word, _ in sorted(counts, key=lambda item: (-item[1], item[0]))]
    lookup = {word: i for i, word in enumerate(vocab)}
    examples = lines.flatMap(lambda line: pairs(line, lookup, window))
    total = examples.count()
    if total > max_pairs:
        examples = examples.sample(False, min(1.0, 1.1 * max_pairs / total), seed=123)
        examples = (examples.zipWithIndex().filter(lambda item: item[1] < max_pairs)
                    .keys())
    examples = examples.cache()
    count = examples.count()
    if count == 0:
        raise ValueError("No word-context pairs found")
    rng = np.random.default_rng(123)
    inputs = rng.normal(0, 0.1, (len(vocab), dim))
    outputs = rng.normal(0, 0.1, inputs.shape)
    for epoch in range(epochs):
        input_bc, output_bc = sc.broadcast(inputs), sc.broadcast(outputs)
        gradients = examples.mapPartitionsWithIndex(
            lambda index, iterator: partition_gradients(
                iterator, input_bc.value, output_bc.value,
                123 + epoch * 1000 + index, negatives))
        for (which, index), gradient in gradients.reduceByKey(
                lambda a, b: a + b).collect():
            (inputs if which == 0 else outputs)[index] -= lr * gradient / count
        input_bc.unpersist()
        output_bc.unpersist()
    sc.stop()
    return vocab, inputs


def show_embeddings(vocab, vectors, plot_dir):
    target_words = ["رستم", "سهراب", "اسفندیار", "رخش", "زال"]
    selected = [(word, vocab.index(word)) for word in target_words if word in vocab]
    for word, index in selected:
        print(word, np.round(vectors[index], 4).tolist())
    if len(selected) < 2 or not plot_dir:
        return
    import matplotlib.pyplot as plt
    target = np.stack([vectors[index] for _, index in selected])
    normalized = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
    similarity = normalized @ normalized.T
    root = Path(plot_dir)
    root.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    image = ax.imshow(similarity, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(selected)), [w for w, _ in selected])
    ax.set_yticks(range(len(selected)), [w for w, _ in selected])
    fig.colorbar(image)
    fig.savefig(root / "similarity.png", bbox_inches="tight")
    plt.close(fig)
    centered = target - target.mean(axis=0)
    projected = centered @ np.linalg.svd(centered, full_matrices=False)[2][:2].T
    fig, ax = plt.subplots()
    ax.scatter(projected[:, 0], projected[:, 1])
    for (word, _), xy in zip(selected, projected):
        ax.annotate(word, xy)
    fig.savefig(root / "pca.png", bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--backend", choices=["local", "spark"], default="local")
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--max-pairs", type=int, default=30000)
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--negatives", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--plot-dir")
    a = p.parse_args()
    if min(a.window, a.max_pairs, a.dim, a.epochs) < 1 or a.negatives < 0 or a.lr <= 0:
        p.error("window, max-pairs, dim, epochs and lr must be positive")
    if a.backend == "spark":
        vocab, vectors = train_spark(a.text, a.window, a.max_pairs, a.dim,
                                      a.epochs, a.negatives, a.lr)
    else:
        vocab, examples = load_local(a.text, a.window, a.max_pairs)
        if not examples:
            p.error("No word-context pairs found")
        vectors = train_local(vocab, examples, a.dim, a.epochs,
                              a.negatives, a.lr)
    print(f"vocabulary={len(vocab)} embedding_dim={a.dim}")
    show_embeddings(vocab, vectors, a.plot_dir)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
