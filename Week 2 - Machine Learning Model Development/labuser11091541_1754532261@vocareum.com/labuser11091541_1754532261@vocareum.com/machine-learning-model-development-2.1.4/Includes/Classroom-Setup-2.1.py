# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

@DBAcademyHelper.add_init
def create_features_table(self):
    from databricks.feature_engineering import FeatureEngineeringClient
    import pandas as pd
    from pyspark.sql.functions import monotonically_increasing_id, col
    
    fe = FeatureEngineeringClient()

    table_name = 'diabetes'

    shared_volume_name = "cdc-diabetes"
    csv_name = "diabetes_binary_5050split_BRFSS2015"
    
    # Define active catalog and schema
    spark.sql(f"USE CATALOG {DA.catalog_name}")
    spark.sql(f"USE {DA.schema_name}")
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")

    # Full path to table in Unity Catalog
    full_table_path = f"{DA.catalog_name}.{DA.schema_name}.{table_name}"

    # Path to CSV file
    dataset_path = f"{DA.paths.datasets.cdc_diabetes}/{shared_volume_name}/{csv_name}.csv"

    # Read the dataset
    df = (
        spark
        .read
        .format("csv")
        .option("header", "true")
        .load(dataset_path)
        .withColumn("unique_id", monotonically_increasing_id()) # Add unique_id column
        )
    
    # Create feature table
    fe.create_table(
        name=full_table_path,
        primary_keys=["unique_id"],
        df=df,
        description="Diabetes Feature Table",
        tags={"source": "silver", "format": "delta"}
    )

# COMMAND ----------

# Initialize DBAcademyHelper
DA = DBAcademyHelper() 
DA.init()