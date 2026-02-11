
"""
Hudi + Lance File Format Demo: Image Similarity Search
(Oxford-IIIT Pet)
========================================================
End-to-end demo:
1) Load Oxford-IIIT Pet (cats & dogs, real photos)
2) Generate embeddings with a timm backbone
3) Store rows in a Hudi table using Lance as the file format
4) Cosine-similarity search using Hudi vector search
5) Save & log the query image, each top-K neighbor, and a combined panel
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    BinaryType,
    ArrayType,
    FloatType,
    IntegerType,
)

import os
from pathlib import Path
import io
import numpy as np

import torch
import timm
from sklearn.preprocessing import normalize
from PIL import Image

# Use headless-friendly backend BEFORE importing pyplot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# torchvision after matplotlib config
from torchvision.datasets import OxfordIIITPet  # noqa: E402


# ======================================================
# CONFIGURATION
# ======================================================

CONFIG = {
    "dataset": "OxfordIIITPet",
    "table_path": "/tmp/hudi_lance_pets",
    "table_name": "pets_lance",
    "n_samples": 1000,  # up to ~7k; keep moderate for quick runs
    "top_k": 5,
    # fast & decent for demo; try convnext_tiny for a bump
    "embedding_model": "mobilenetv3_small_100",
    "output_dir": os.getenv("HUDI_LANCE_DEMO_OUTDIR", "./outputs"),
    "panel_filename": "hudi_lance_results.png",
    "log_level": "ERROR",  # WARN or ERROR to reduce noise
    "hide_progress": True,  # hide Spark console progress bar
}


# ======================================================
# UTILITIES
# ======================================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_png_bytes(img_bytes: bytes, path: Path):
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        f.write(img_bytes)


# ======================================================
# 1. SPARK SESSION SETUP
# ======================================================

def create_spark():
    """Initialize Spark with Hudi + Lance support."""
    builder = (
        SparkSession.builder.appName("Hudi-Lance-Demo-Pets")
        .config(
            "spark.serializer",
            "org.apache.spark.serializer.KryoSerializer",
        )
        .config(
            "spark.sql.extensions",
            "org.apache.spark.sql.hudi.HoodieSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.hudi.catalog.HoodieCatalog",
        )
    )

    if CONFIG.get("hide_progress", True):
        builder = builder.config("spark.ui.showConsoleProgress", "false")

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(CONFIG.get("log_level", "ERROR"))
    return spark


# ======================================================
# 2. LOAD DATASET (Oxford-IIIT Pet)
# ======================================================

def load_dataset(n_samples):
    """
    Load Oxford-IIIT Pet (trainval split) and return:
    - data: list[dict] with image bytes + metadata
    - class_names: list[str] of breed names
    """
    print(f"Loading dataset: Oxford-IIIT Pet ({n_samples} samples)...")

    root = "~/.cache/torchvision"
    ds = OxfordIIITPet(root=root, split="trainval", download=True)
    # torchvision exposes .classes
    class_names = ds.classes

    # Random subset
    rng = np.random.default_rng()
    n = min(n_samples, len(ds))
    indices = rng.choice(len(ds), size=n, replace=False)

    data = []
    for idx in indices:
        img, label = ds[idx]  # img: PIL, label: int (0..36)
        img = img.convert("RGB")  # ensure RGB

        # Serialize as PNG bytes
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        img_bytes = bio.getvalue()

        w, h = img.size
        category = class_names[label] if isinstance(class_names, list) else str(label)
        safe_category = category.replace("/", "_")  # sanitize for partitions

        data.append(
            {
                "image_id": f"pets_{idx:06d}",
                "category": category,  # original label
                "category_sanitized": safe_category,  # safe for partitions
                "label": int(label),
                "description": f"{category} from Oxford-IIIT Pet",
                "image_bytes": img_bytes,
                "width": int(w),
                "height": int(h),
            }
        )

    print(f"✓ Loaded {len(data)} images from Oxford-IIIT Pet")
    return data, class_names


# ======================================================
# 3. EMBEDDING MODEL (timm)
# ======================================================

def create_embedding_model():
    """
    Create embedding model using timm (PyTorch Image Models).
    num_classes=0 returns penultimate features (embeddings).
    """
    print(f"Loading embedding model: {CONFIG['embedding_model']}...")
    model = timm.create_model(
        CONFIG["embedding_model"],
        pretrained=True,
        num_classes=0,
    )
    model.eval()

    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(
        **data_config,
        is_training=False,
    )
    print("✓ Model loaded")
    return model, transform


def generate_embeddings(data, model, transform):
    """
    Generate embeddings for all images using batched inference and L2-normalize.
    """
    print(f"Generating embeddings for {len(data)} images...")

    images = []
    for item in data:
        img = Image.open(io.BytesIO(item["image_bytes"])).convert("RGB")
        images.append(transform(img))

    batch = torch.stack(images)

    with torch.no_grad():
        feats = model(batch).detach().cpu().numpy()

    # L2-normalize for cosine similarity stability
    feats = normalize(feats)

    # Attach back
    for i, item in enumerate(data):
        item["embedding"] = feats[i].tolist()

    print(f"✓ Generated embeddings (dimension: {feats.shape[1]})")
    return data


# ======================================================
# 4. WRITE TO HUDI TABLE WITH LANCE FORMAT
# ======================================================

def write_to_hudi(spark, data):
    """Write data to Hudi table with Lance file format."""
    print("\nWriting to Hudi table with Lance format...")

    schema = StructType(
        [
            StructField("image_id", StringType(), False),
            StructField("category", StringType(), False),
            StructField("category_sanitized", StringType(), False),
            StructField("label", IntegerType(), False),
            StructField("description", StringType(), True),
            StructField("image_bytes", BinaryType(), False),
            StructField("width", IntegerType(), False),
            StructField("height", IntegerType(), False),
            StructField("embedding", ArrayType(FloatType()), False),
        ]
    )

    df = spark.createDataFrame(data, schema=schema)

    # Hudi write options with Lance format
    hudi_options = {
        "hoodie.table.name": CONFIG["table_name"],
        "hoodie.datasource.write.recordkey.field": "image_id",
        "hoodie.datasource.write.precombine.field": "image_id",
        "hoodie.datasource.write.partitionpath.field": "category_sanitized",
        "hoodie.datasource.write.table.type": "COPY_ON_WRITE",
        "hoodie.datasource.write.operation": "upsert",
        # LANCE FILE FORMAT
        "hoodie.table.base.file.format": "lance",
        "hoodie.write.record.merge.custom.implementation.classes": (
            "org.apache.hudi.DefaultSparkRecordMerger"
        )
    }

    (
        df.write.format("hudi")
        .options(**hudi_options)
        .mode("overwrite")
        .save(CONFIG["table_path"])
    )

    count = df.count()
    print(f"✓ Wrote {count} records to Hudi table")
    return df


# ======================================================
# 5. SIMILARITY SEARCH USING HUDI VECTOR SEARCH
# ======================================================

def find_similar(spark, query_embedding):
    """
    Find similar images using Hudi vector search.
    Returns result rows with image bytes & similarity.
    """
    print(f"\nPerforming similarity search (top {CONFIG['top_k']})...")

    # Create a 1-row DataFrame with the query embedding
    query_schema = StructType(
        [StructField("query_embedding", ArrayType(FloatType()), False)]
    )
    query_df = spark.createDataFrame([(query_embedding,)], schema=query_schema)

    # Register as temp view
    query_df.createOrReplaceTempView("query_vector")

    # Use scalar subquery to extract the embedding
    query = f"""
        SELECT image_id, category, image_bytes, _distance
        FROM hudi_vector_search(
            '{CONFIG['table_path']}',
            'embedding',
            (SELECT query_embedding FROM query_vector),
            {CONFIG['top_k'] + 1},
            'cosine'
        )
        ORDER BY _distance
    """

    df = spark.sql(query)
    rows = df.collect()

    # Convert results, excluding the query image (distance ~0)
    # For cosine distance with L2-normalized embeddings:
    #   distance = 1 - similarity, so similarity = 1 - distance
    results = []
    for row in rows:
        distance = float(row["_distance"])

        # Skip the query image itself (distance very close to 0)
        if distance < 0.001:
            continue

        if len(results) >= CONFIG["top_k"]:
            break

        similarity = 1.0 - distance
        results.append(
            {
                "image_id": row["image_id"],
                "category": row["category"],
                "image_bytes": row["image_bytes"],
                "similarity": similarity,
            }
        )

    print(f"✓ Found {len(results)} similar images using Hudi vector search")
    return results


# ======================================================
# 6. VISUALIZATION & PER-IMAGE SAVES
# ======================================================

def visualize_and_save(query_image_bytes, query_category, results):
    """
    Save query & result images individually, and create a combined panel figure.
    Returns dict with saved file paths.
    """
    print("\nCreating visualization and saving images...")

    out_dir = Path(CONFIG["output_dir"])
    ensure_dir(out_dir)

    # Save individual images
    query_path = out_dir / "query.png"
    save_png_bytes(query_image_bytes, query_path)

    result_paths = []
    for i, r in enumerate(results, 1):
        p = out_dir / f"top{i}.png"
        save_png_bytes(r["image_bytes"], p)
        result_paths.append(str(p.resolve()))

    # Combined panel
    n_results = len(results)
    fig, axes = plt.subplots(1, n_results + 1, figsize=(3 * (n_results + 1), 3.2))

    # Query image
    query_img = Image.open(io.BytesIO(query_image_bytes)).convert("RGB")
    axes[0].imshow(query_img)
    axes[0].set_title(f"QUERY\n{query_category}", fontweight="bold")
    axes[0].axis("off")

    # Similar images
    for i, result in enumerate(results):
        img = Image.open(io.BytesIO(result["image_bytes"])).convert("RGB")
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(
            f"{result['category']}\nSim: {result['similarity']:.3f}"
        )
        axes[i + 1].axis("off")

    plt.tight_layout()

    panel_path = out_dir / CONFIG["panel_filename"]
    plt.savefig(str(panel_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("✓ Saved files:")
    print(f"  Query image: {query_path.resolve()}")
    for i, p in enumerate(result_paths, 1):
        print(f"  Top{i}: {p}")
    print(f"  Panel: {panel_path.resolve()}")

    return {
        "query": str(query_path.resolve()),
        "tops": result_paths,
        "panel": str(panel_path.resolve()),
    }


# ======================================================
# MAIN EXECUTION
# ======================================================

def main():
    print("\n" + "=" * 80)
    print("HUDI + LANCE: IMAGE SIMILARITY SEARCH DEMO (Oxford-IIIT Pet)")
    print("=" * 80 + "\n")

    spark = create_spark()

    # 1) Load dataset
    data, class_names = load_dataset(CONFIG["n_samples"])

    # 2) Embeddings
    model, transform = create_embedding_model()
    data = generate_embeddings(data, model, transform)

    # 3) Write to Hudi with Lance
    df = write_to_hudi(spark, data)

    print("\nSample rows:")
    df.select("image_id", "category", "description").show(5, truncate=False)

    # 4) Pick a random query
    query_idx = np.random.randint(len(data))
    query_item = data[query_idx]
    print(f"\nQuery: {query_item['image_id']} ({query_item['category']})")

    results = find_similar(spark, query_item["embedding"])

    # 5) Print neighbors
    print("\nTop matches:")
    for i, r in enumerate(results, 1):
        print(
            f"  {i}. {r['image_id']} - {r['category']} "
            f"(similarity: {r['similarity']:.3f})"
        )

    # 6) Save images & panel
    saved = visualize_and_save(
        query_item["image_bytes"],
        query_item["category"],
        results,
    )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Dataset: {CONFIG['dataset']} ({CONFIG['n_samples']} images)")
    print(f"✓ Model: {CONFIG['embedding_model']}")
    print("✓ Storage: Hudi + Lance file format")
    print("✓ Search: Hudi vector search (cosine distance, L2-normalized embeddings)")
    print("✓ Files:")
    print(f"    Query: {saved['query']}")
    for i, p in enumerate(saved["tops"], 1):
        print(f"    Top{i}: {p}")
    print(f"    Panel: {saved['panel']}")
    print("=" * 80 + "\n")

    spark.stop()


if __name__ == "__main__":
    main()
