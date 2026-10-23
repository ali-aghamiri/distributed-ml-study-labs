"""Annual-income statistics and customer K-means with optional Spark ML."""
import argparse
import csv

INCOME = "Annual Income (k$)"
SPENDING = "Spending Score (1-100)"


def local(path, clusters):
    import numpy as np
    from sklearn.cluster import KMeans
    with open(path, newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    values = np.array([[float(row[INCOME]), float(row[SPENDING])] for row in rows])
    income = values[:, 0]
    labels = KMeans(n_clusters=clusters, random_state=42, n_init=10).fit_predict(values)
    print(f"count={len(rows)} income_min={income.min():.2f} "
          f"income_max={income.max():.2f} income_mean={income.mean():.2f} "
          f"income_variance={income.var():.2f}")
    print("cluster_sizes=", np.bincount(labels, minlength=clusters).tolist())


def spark(path, clusters):
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.feature import VectorAssembler
    session = SparkSession.builder.appName("DMLS customer clustering").getOrCreate()
    frame = session.read.csv(path, header=True, inferSchema=True)
    frame.select(F.min(INCOME), F.max(INCOME), F.avg(INCOME),
                 F.variance(INCOME)).show()
    features = VectorAssembler(inputCols=[INCOME, SPENDING], outputCol="features")
    fitted = KMeans(k=clusters, seed=42, featuresCol="features").fit(
        features.transform(frame))
    fitted.transform(features.transform(frame)).groupBy("prediction").count().show()
    session.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--clusters", type=int, default=5)
    p.add_argument("--backend", choices=["local", "spark"], default="local")
    a = p.parse_args()
    if a.clusters < 1:
        p.error("clusters must be positive")
    (spark if a.backend == "spark" else local)(a.csv, a.clusters)
