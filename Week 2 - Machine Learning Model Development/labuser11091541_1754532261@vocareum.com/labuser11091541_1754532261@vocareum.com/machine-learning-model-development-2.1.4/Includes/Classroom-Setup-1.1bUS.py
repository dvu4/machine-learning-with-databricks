# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
np.set_printoptions(precision=2)

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# COMMAND ----------

import os

os.environ["OMP_NUM_THREADS"] = "1"  # Restrict OpenMP threads
os.environ["MKL_NUM_THREADS"] = "1"  # Restrict MKL threads
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Restrict OpenBLAS threads
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # Restrict NumExpr threads
import threadpoolctl
threadpoolctl.threadpool_limits(limits=1)

# COMMAND ----------

@DBAcademyHelper.add_init
def create_news_features_table(self):
    from datasets import load_dataset
    import pandas as pd

    # Define active catalog and schema
    spark.sql(f"USE CATALOG {DA.catalog_name}")
    spark.sql(f"USE {DA.schema_name}")

    # Load AG News Dataset (Limit to 1000 rows)
    ds = load_dataset("wangrongsheng/ag_news", split="train").select(range(1000))

    # Convert dataset to Pandas DataFrame
    news_df = pd.DataFrame(ds)

    # Convert Pandas DataFrame to Spark DataFrame
    news_spark_df = spark.createDataFrame(news_df)

    # Define table path
    table_path = f"{DA.catalog_name}.{DA.schema_name}.ag_news_table"

    # Save dataset as a Delta table
    news_spark_df.write.format("delta").mode("overwrite").saveAsTable(table_path)

    print(f"Dataset saved at {table_path}")

    from databricks.feature_engineering import FeatureEngineeringClient
    from pyspark.sql.functions import monotonically_increasing_id

    fe = FeatureEngineeringClient()
    
    # Define active catalog and schema
    spark.sql(f"USE CATALOG {DA.catalog_name}")
    spark.sql(f"USE {DA.schema_name}")
    
    # Drop table if it exists to ensure a fresh start
    spark.sql(f"DROP TABLE IF EXISTS {DA.catalog_name}.{DA.schema_name}.ag_news_features")
    
    # Read dataset from Delta table
    news_spark_df = spark.table(table_path)
    
    # Drop label column for unsupervised learning
    news_spark_df = news_spark_df.drop("label")

    # Add unique_id column for feature store indexing
    news_spark_df = news_spark_df.withColumn("unique_id", monotonically_increasing_id())

    # Define feature table name
    table_name = f"{DA.catalog_name}.{DA.schema_name}.ag_news_features"

    # Create the feature table
    fe.create_table(
        name=table_name,
        primary_keys=["unique_id"],
        df=news_spark_df,
        description="AG News Feature Table for Unsupervised Learning",
        tags={"source": "silver", "format": "delta"}
    )
    
    print(f"Feature table created: {table_name}")

# COMMAND ----------

# Initialize DBAcademyHelper
DA = DBAcademyHelper() 
DA.init()