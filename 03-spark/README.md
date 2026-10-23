# Lab 03 — Spark text analysis, skip-gram and K-means

The course archive contains Shahnameh text and `customers.csv`; place them under a local `data/` directory (ignored by Git). `text_analysis.py` counts verse lines, unique words, the final-word proxy for rhyme, and trigrams. `skipgram.py` implements negative-sampling skip-gram updates; Spark mode distributes gradient computation over RDD partitions, then synchronizes updated weights each epoch. `customers_kmeans.py` uses Spark ML's `KMeans` and computes the requested income statistics.

```powershell
python 03-spark/text_analysis.py --text data/shahname.txt --backend local
python 03-spark/skipgram.py --text data/shahname.txt --backend local --max-pairs 3000
python 03-spark/customers_kmeans.py --csv data/customers.csv --backend local
```

Install PySpark and a compatible Java runtime for `--backend spark`. Use `spark-submit` for a cluster run. To use the course HDFS installation, pass an HDFS URI as `--text` or `--csv` in Spark mode. Local mode is a reproducible logic check, not a Spark or HDFS benchmark. The skip-gram implementation is intentionally small and uses global model synchronization rather than a production word2vec optimizer.
