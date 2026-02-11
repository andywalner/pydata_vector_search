"""
Hudi + Lance Demo: Intelligent Recruitment Platform
(Hybrid Search + Analytics on the Lakehouse)
===================================================
Flow:
1. Ingest Job Postings (Structured + Unstructured Data)
2. User "Uploads" a Resume (Vector Search)
3. Apply Business Rules (Hybrid Search: Vector + SQL Filters)
4. Show Executive Dashboard (Analytics on the same data)
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_unixtime, count, avg, date_format
from pyspark.sql.types import *
import shutil
import os
import random
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# ======================================================
# CONFIGURATION
# ======================================================
CONFIG = {
    "table_path": "/tmp/hudi_recruiting_lake",
    "table_name": "job_market",
    "embedding_model": "all-MiniLM-L6-v2",
    "clean_start": True
}

# ======================================================
# 1. SETUP & DATA GENERATION
# ======================================================

def create_spark():
    return (SparkSession.builder.appName("Recruiting-Lakehouse")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.sql.extensions", "org.apache.spark.sql.hudi.HoodieSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.hudi.catalog.HoodieCatalog")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())

def generate_realistic_jobs():
    """Generates rich data: Text for AI, Numbers for BI."""
    print("Generating 100 job postings with historical dates...")
    
    titles = [
        ("Senior Data Scientist", "Build LLM applications and predictive models. Python, PyTorch.", "Tech"),
        ("Marketing Director", "Lead global brand strategy and social media campaigns.", "Marketing"),
        ("Solutions Architect", "Design cloud infrastructure on AWS and Azure for enterprise clients.", "Tech"),
        ("HR Business Partner", "Manage employee relations and internal hiring strategy.", "HR"),
        ("Frontend Developer", "React and TypeScript expert needed for high-traffic e-commerce site.", "Tech"),
        ("Sales Executive", "B2B enterprise sales. High commission. Travel required.", "Sales")
    ]
    
    cities = ["New York", "San Francisco", "Austin", "London", "Remote"]
    
    data = []
    base_date = datetime.now()
    
    for i in range(100):
        t_info = random.choice(titles)
        city = random.choice(cities)
        # Salary varies by city/role logic (simplified)
        base_sal = 160000 if t_info[2] == "Tech" else 90000
        salary = int(base_sal * random.uniform(0.8, 1.2))
        
        # Date distribution (last 6 months)
        post_date = base_date - timedelta(days=random.randint(0, 180))
        
        data.append({
            "job_id": f"job_{i:03d}",
            "title": t_info[0],
            "description": t_info[1] + f" Located in {city}.",
            "department": t_info[2],
            "location": city,
            "salary": salary,
            "posted_at": int(post_date.timestamp()), # For time-series analytics
            "text_for_vector": f"{t_info[0]} {t_info[1]}"
        })
        
    return data

# ======================================================
# 2. INGESTION (THE "LAKEHOUSE" FOUNDATION)
# ======================================================

def ingest_data(spark, data):
    # 1. Embed Descriptions
    model = SentenceTransformer(CONFIG['embedding_model'])
    embeddings = model.encode([r['text_for_vector'] for r in data])
    
    for i, row in enumerate(data):
        row['embedding'] = embeddings[i].tolist()

    # 2. Define Schema
    schema = StructType([
        StructField("job_id", StringType(), False),
        StructField("title", StringType(), False),
        StructField("description", StringType(), False),
        StructField("department", StringType(), False),
        StructField("location", StringType(), False),
        StructField("salary", IntegerType(), False),
        StructField("posted_at", LongType(), False), # Timestamp
        StructField("text_for_vector", StringType(), False),
        StructField("embedding", ArrayType(FloatType()), False),
    ])

    # 3. Write to Hudi (Lance Format)
    if CONFIG["clean_start"] and os.path.exists(CONFIG["table_path"]):
        shutil.rmtree(CONFIG["table_path"])

    df = spark.createDataFrame(data, schema=schema)
    
    hudi_options = {
        "hoodie.table.name": CONFIG["table_name"],
        "hoodie.datasource.write.recordkey.field": "job_id",
        "hoodie.datasource.write.partitionpath.field": "department",
        "hoodie.datasource.write.table.type": "COPY_ON_WRITE",
        "hoodie.datasource.write.operation": "upsert",
        "hoodie.table.base.file.format": "lance",
        "hoodie.write.record.merge.custom.implementation.classes": "org.apache.hudi.DefaultSparkRecordMerger"
    }

    df.write.format("hudi").options(**hudi_options).mode("overwrite").save(CONFIG["table_path"])
    print(f"✓ Ingested {len(data)} jobs into the Lakehouse.")
    return model

# ======================================================
# 3. THE DEMO: RESUME MATCHING
# ======================================================

def demo_resume_matching(spark, model):
    print("\n" + "="*50)
    print("DEMO PART 1: The 'Smart' Candidate Match")
    print("="*50)
    
    # Simulate a Resume Upload
    resume_text = """
    EXPERIENCE:
    - 5 years building Machine Learning models using Python and Scikit-Learn.
    - Deployed Large Language Models (LLMs) to production.
    - Strong background in backend engineering and API design.
    """
    print(f"📄 User Resume Uploaded: \n{resume_text.strip()}\n")
    
    # Vectorize Resume
    resume_vector = model.encode([resume_text])[0].tolist()
    
    # Register Query Vector
    spark.createDataFrame([(resume_vector,)], ["q_vec"]).createOrReplaceTempView("query_input")
    
    # --- SCENARIO A: Pure Vector Search ---
    print("🔎 Executing Vector Search (Semantic Match)...")
    matches = spark.sql(f"""
        SELECT title, location, salary, (1 - _distance) as score
        FROM hudi_vector_search(
            '{CONFIG['table_path']}', 'embedding', (SELECT q_vec FROM query_input), 5, 'cosine'
        )
    """).collect()
    
    print("\nTop Matches for your Resume:")
    for row in matches:
        print(f"  • {row.title} ({row.location}) - Match Score: {row.score:.2f}")

    # --- SCENARIO B: Hybrid Search (The Business Requirement) ---
    print("\n⚠️  User Feedback: 'I only want Remote jobs paying > $150k'")
    print("🔎 Executing Hybrid Search (Vector + SQL Filters)...")
    
    # Note: In a real Hudi implementation, you filter the results of the vector search
    # or push predicates down depending on the exact Hudi version support.
    # Here we simulate the logical flow of a Hybrid Query.
    
    hybrid_query = f"""
        SELECT * FROM (
            SELECT title, location, salary, (1 - _distance) as score
            FROM hudi_vector_search(
                '{CONFIG['table_path']}', 'embedding', (SELECT q_vec FROM query_input), 20, 'cosine'
            )
        ) 
        WHERE location = 'Remote' AND salary > 150000
        ORDER BY score DESC
        LIMIT 5
    """
    hybrid_matches = spark.sql(hybrid_query).collect()
    
    print("\nTop HYBRID Matches:")
    if not hybrid_matches:
        print("  (No matches found with these strict constraints - this is also a valuable insight!)")
    for row in hybrid_matches:
        print(f"  • {row.title} [${row.salary:,}] - {row.location}")


# ======================================================
# 4. THE DEMO: ANALYTICS DASHBOARD
# ======================================================

def demo_analytics_dashboard(spark):
    print("\n" + "="*50)
    print("DEMO PART 2: The Executive Dashboard")
    print("Value: The SAME data matches resumes AND powers BI.")
    print("="*50)
    
    spark.read.format("hudi").load(CONFIG["table_path"]).createOrReplaceTempView("jobs_table")
    
    # 1. Time Series Data (Job Postings per Month)
    print("Generating 'Job Trends' Chart...")
    trend_df = spark.sql("""
        SELECT from_unixtime(posted_at, 'yyyy-MM') as month, count(*) as count
        FROM jobs_table
        GROUP BY month ORDER BY month
    """).toPandas()
    
    # 2. Salary Analysis by Dept
    print("Generating 'Salary Insights' Chart...")
    salary_df = spark.sql("""
        SELECT department, avg(salary) as avg_salary
        FROM jobs_table GROUP BY department
    """).toPandas()
    
    # PLOTTING
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Trend
    axes[0].plot(trend_df['month'], trend_df['count'], marker='o', color='green')
    axes[0].set_title("Market Demand: Job Postings Over Time")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("New Jobs")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Salary
    axes[1].bar(salary_df['department'], salary_df['avg_salary'], color='skyblue')
    axes[1].set_title("Compensation Benchmark by Dept")
    axes[1].set_ylabel("Avg Salary ($)")
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Dashboard generated from Hudi table.")
    print("  (In a real app, this would be a live Streamlit/Tableau view)")

# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    spark = create_spark()
    
    # 1. Setup
    jobs_data = generate_realistic_jobs()
    model = ingest_data(spark, jobs_data)
    
    # 2. Run Demo
    demo_resume_matching(spark, model)
    demo_analytics_dashboard(spark)
    
    spark.stop()
