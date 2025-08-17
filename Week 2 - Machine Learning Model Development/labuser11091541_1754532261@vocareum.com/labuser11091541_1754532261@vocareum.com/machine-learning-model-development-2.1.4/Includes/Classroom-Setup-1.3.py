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



# COMMAND ----------

@DBAcademyHelper.add_init
def create_features_table(self):

    from pyspark.sql.functions import monotonically_increasing_id, col
    from pyspark.ml.feature import StringIndexer
    from pyspark.ml import Pipeline

    from databricks.feature_engineering import FeatureEngineeringClient    

    import mlflow
    
    table_name = 'telco'

    shared_volume_name = "telco"
    csv_name = "telco-customer-churn"

    # Full path to table in Unity Catalog
    full_table_path = f"{DA.catalog_name}.{DA.schema_name}.{table_name}"

    # Path to CSV file
    dataset_path = f"{DA.paths.datasets.telco}/{shared_volume_name}/{csv_name}.csv"

    # Define active catalog and schema
    spark.sql(f"USE CATALOG {DA.catalog_name}")
    spark.sql(f"USE {DA.schema_name}")
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")
    

    # disable autologging 
    mlflow.autolog(disable=True)

    # Read the CSV file into a Spark DataFrame
    feature_data_spark = (
        spark
        .read
        .format('csv')
        .option('header', True)
        .load(dataset_path)
        )

    # Fill missing values with 0
    feature_data_spark = feature_data_spark.fillna(0)

    # Identify categorical columns
    categorical_cols = [col_name for col_name, data_type in feature_data_spark.dtypes if data_type == 'string']

    # Create a StringIndexer for each categorical column
    indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="skip") for col in categorical_cols]

    # Create a pipeline to apply all indexers
    pipeline = Pipeline(stages=indexers)

    # Fit the pipeline to the data
    feature_data_spark = pipeline.fit(feature_data_spark).transform(feature_data_spark)

    # Drop the original categorical columns
    feature_data_spark = feature_data_spark.drop(*categorical_cols)

    # Rename the indexed columns to match the original column names
    for col in categorical_cols:
        feature_data_spark = feature_data_spark.withColumnRenamed(f"{col}_index", col)
    
     # Create the feature table using the PySpark DataFrame
    fe = FeatureEngineeringClient()
    fe.create_table(
        name=full_table_path,
        primary_keys=["customerID"],
        df=feature_data_spark,
        description="Telco Dataset",
        tags={"source": "silver", "format": "delta"}
    )

# COMMAND ----------

# Initialize DBAcademyHelper
DA = DBAcademyHelper() 
DA.init()