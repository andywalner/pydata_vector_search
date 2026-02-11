# Hudi + Lance: Vector Search on the Lakehouse

A live demo for PyData showing how Apache Hudi with the Lance file format enables **vector search**, **hybrid search**, and **analytics** on a single lakehouse table — no data copying, no separate vector database.

## What This Demo Shows

We build an intelligent job-matching platform. One Hudi table stores structured data (salary, location, department) **and** vector embeddings side by side. Then we query it three different ways.

### 1. Resume Upload (Vector Search)

A user uploads a technical Python/ML resume. It never says "Senior Data Scientist" — it describes skills like "deployed LLMs to production" and "Scikit-Learn." The embedding model finds `Senior Data Scientist` anyway because the **meaning** matches. This is vector search.

### 2. Business Constraints (Hybrid Search)

The user says: *"I can't relocate, and I need at least $150k."* We combine the same vector search with standard SQL `WHERE` clauses — `location = 'Remote' AND salary > 150000` — against the same Hudi table. Vector + SQL filters in one query. This is hybrid search.

### 3. Executive Dashboard (Analytics)

Now we switch hats. We're an analyst on the job platform team. We don't care about individual matches — we care about hiring velocity and salary bands across departments. We query the **exact same dataset** to build time-series and compensation charts. No ETL to a separate warehouse.

## Setup

### 1. Build Hudi with Lance + Vector Search support

This demo runs on a feature branch that adds Lance file format and vector search to Hudi.

```bash
git clone https://github.com/rahil-c/hudi.git
cd hudi
git checkout rahil/rfc100-hudi-vector-search
mvn clean package -DskipTests -pl hudi-spark-datasource/hudi-spark3.5-bundle -am
```

The built JAR will be at:
```
hudi-spark-datasource/hudi-spark3.5-bundle/target/hudi-spark3.5-bundle_2.12-*.jar
```

### 2. Python dependencies

```bash
pip install pyspark pandas matplotlib sentence-transformers
```

### 3. Launch

Start Spark with the Hudi bundle JAR:

```bash
pyspark --jars /path/to/hudi-spark3.5-bundle_2.12-*.jar
```

Or run the notebook:

```bash
jupyter notebook demo.ipynb
```

(Make sure `PYSPARK_SUBMIT_ARGS` includes `--jars /path/to/hudi-spark3.5-bundle_2.12-*.jar` or configure it in the notebook's Spark session.)

## Repo Structure

```
demo.py          # Standalone script
demo.ipynb       # Jupyter notebook (same content, cell-by-cell)
```
