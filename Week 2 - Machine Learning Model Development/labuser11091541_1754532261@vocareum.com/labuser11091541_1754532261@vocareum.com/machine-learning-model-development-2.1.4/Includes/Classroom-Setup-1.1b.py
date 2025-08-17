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

@DBAcademyHelper.add_init
def create_features_table(self):

    from databricks.feature_engineering import FeatureEngineeringClient
    import pandas as pd
    from pyspark.sql.functions import monotonically_increasing_id, col
    
    fe = FeatureEngineeringClient()
    
    # define active catalog and schema
    spark.sql(f"USE CATALOG {DA.catalog_name}")
    spark.sql(f"USE {DA.schema_name}")
    spark.sql(f"DROP TABLE IF EXISTS diabetes_binary")

    # Set the path of the dataset
    shared_volume_name = 'cdc-diabetes' # From Marketplace
    csv_name = 'diabetes_binary_5050split_BRFSS2015' # CSV file name
    dataset_path = f"{DA.paths.datasets.cdc_diabetes}/{shared_volume_name}/{csv_name}.csv" # Full path


    # Read the dataset
    df = spark.read.format("csv").option("header", "true").load(dataset_path)
    df = df.withColumn("unique_id", monotonically_increasing_id())   # Add unique_id column

    # create the feature table using the PySpark DataFrame
    table_name = f"{DA.catalog_name}.{DA.schema_name}.diabetes_binary"
    fe.create_table(
        name=table_name,
        primary_keys=["unique_id"],
        df=df,
        description="Diabetes Feature Table",
        tags={"source": "silver", "format": "delta"}
    )

# COMMAND ----------

# Initialize DBAcademyHelper
DA = DBAcademyHelper() 
DA.init()