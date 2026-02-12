# Hudi + Lance: Vector Search on the Lakehouse

A live demo for PyData showing how Apache Hudi with the Lance file format enables **vector search**, **hybrid search**, and **analytics** on a single lakehouse table — no data copying, no separate vector database.

## What This Demo Shows

We load ~3k real data science job postings from [HuggingFace](https://huggingface.co/datasets/nathansutton/data-science-job-descriptions) into a single Hudi table — structured fields (company, title) **and** vector embeddings side by side. Then we query it three different ways.

### 1. Resume Upload (Vector Search)

A user uploads a technical Python/ML resume. It never says "Senior Data Scientist" — it describes skills like "deployed LLMs to production" and "Scikit-Learn." The embedding model finds `Senior Data Scientist` anyway because the **meaning** matches. This is vector search.

### 2. Business Constraints (Hybrid Search)

The user says: *"I specifically want to work at Reddit."* We combine the same vector search with a standard SQL `WHERE company = 'Reddit'` clause against the same Hudi table. Vector + SQL filters in one query. This is hybrid search.

### 3. Executive Dashboard (Analytics)

Now we switch hats. We're an analyst on the job platform team. We don't care about individual matches — we care about which companies are hiring the most and what roles dominate the market. We query the **exact same dataset** to build charts. No ETL to a separate warehouse.

## Prerequisites

- **Java 17** — required by the Hudi vector search branch
  ```bash
  # macOS
  brew install openjdk@17
  export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
  ```

- **Apache Spark 3.5.x** — download the [pre-built package](https://spark.apache.org/downloads.html) (Hadoop 3, Scala 2.12)
  ```bash
  export SPARK_HOME=/path/to/spark-3.5.3-bin-hadoop3
  export PATH=$SPARK_HOME/bin:$PATH
  ```

- **Maven** — for building Hudi from source

- **Lance Spark bundle** — the Lance file format runtime, distributed separately. **Use the Scala 2.12 variant** (not 2.13) to match Spark 3.5.x.
  ```bash
  curl -L -o lance-spark-bundle-3.5_2.12-0.0.14.jar \
    "https://repo1.maven.org/maven2/com/lancedb/lance-spark-bundle-3.5_2.12/0.0.14/lance-spark-bundle-3.5_2.12-0.0.14.jar"
  ```

## Setup

### 1. Build Hudi with Lance + Vector Search support

This demo runs on a feature branch that adds Lance file format and vector search to Hudi.

```bash
git clone https://github.com/rahil-c/hudi.git
cd hudi
git checkout rahil/rfc100-hudi-vector-search
mvn clean package -DskipTests -Dscalastyle.skip=true -Dcheckstyle.skip=true \
    -pl packaging/hudi-spark-bundle -am
```

The built JAR will be at:
```
packaging/hudi-spark-bundle/target/hudi-spark3.5-bundle_2.12-1.2.0-SNAPSHOT.jar
```

### 2. Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch

Both JARs must be on the classpath. Set `PYSPARK_SUBMIT_ARGS` and start Jupyter:

```bash
export PYSPARK_SUBMIT_ARGS="--jars /path/to/hudi-spark3.5-bundle_2.12-1.2.0-SNAPSHOT.jar,/path/to/lance-spark-bundle-3.5_2.12-0.0.14.jar pyspark-shell"
jupyter notebook demo.ipynb
```

Or launch PySpark directly:

```bash
pyspark --jars /path/to/hudi-spark3.5-bundle_2.12-1.2.0-SNAPSHOT.jar,/path/to/lance-spark-bundle-3.5_2.12-0.0.14.jar
```

## Repo Structure

```
demo.ipynb         # Jupyter notebook (the full demo)
requirements.txt   # Python dependencies
```
