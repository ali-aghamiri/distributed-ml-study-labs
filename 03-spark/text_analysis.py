"""Shahnameh line, rhyme and trigram analysis using local or Spark RDDs."""
import argparse
from collections import Counter
import re
import sys


def parse_verse(line):
    if line.lstrip().startswith("#") or not line.strip():
        return None
    halves = line.split("\t")
    if len(halves) < 2:
        return None
    tokens = re.findall(r"[^\W\d_]+", line, flags=re.UNICODE)
    last_half = re.findall(r"[^\W\d_]+", halves[-1], flags=re.UNICODE)
    return (tokens, last_half[-1] if last_half else None) if tokens else None


def trigrams(tokens):
    return [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]


def local(path):
    with open(path, encoding="utf-8-sig") as stream:
        verses = [v for line in stream if (v := parse_verse(line))]
    counts = Counter(tuple(t) for words, _ in verses for t in trigrams(words))
    rhymes = Counter(rhyme for _, rhyme in verses if rhyme)
    print(f"verses={len(verses)} unique_words="
          f"{len(set(w for words, _ in verses for w in words))} "
          f"unique_trigrams={len(counts)}")
    print("top_rhymes=", rhymes.most_common(10))
    print("top_trigrams=", counts.most_common(10))


def spark(path):
    from pyspark import SparkContext
    sc = SparkContext.getOrCreate()
    verses = sc.textFile(path).map(parse_verse).filter(lambda x: x is not None).cache()
    word_count = verses.flatMap(lambda v: v[0]).distinct().count()
    rhyme_counts = (verses.map(lambda v: v[1]).filter(lambda x: x is not None)
                    .map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b))
    tri_counts = (verses.flatMap(lambda v: [(t, 1) for t in trigrams(v[0])])
                  .reduceByKey(lambda a, b: a + b))
    print(f"verses={verses.count()} unique_words={word_count} "
          f"unique_trigrams={tri_counts.count()}")
    print("top_rhymes=", rhyme_counts.takeOrdered(10, key=lambda x: -x[1]))
    print("top_trigrams=", tri_counts.takeOrdered(10, key=lambda x: -x[1]))
    sc.stop()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--backend", choices=["local", "spark"], default="local")
    a = p.parse_args()
    (spark if a.backend == "spark" else local)(a.text)
