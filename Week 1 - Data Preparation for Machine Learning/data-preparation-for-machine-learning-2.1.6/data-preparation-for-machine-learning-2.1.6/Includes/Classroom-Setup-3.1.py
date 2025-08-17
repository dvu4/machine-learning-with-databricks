# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

from pyspark.sql.functions import col

# COMMAND ----------

# Create a delta table for diet features
@DBAcademyHelper.add_init
def create_security_features_table(self):
    
    spark.sql(f"USE CATALOG {DA.catalog_name}")
    spark.sql(f"USE SCHEMA {DA.schema_name}")

    # Load dataset with spark
    shared_volume_name = 'telco' # From Marketplace
    csv_name = 'telco-customer-churn-missing' # CSV file name
    dataset_path = f"{DA.paths.datasets.telco}/{shared_volume_name}/{csv_name}.csv" # Full path

    df = spark.read.csv(dataset_path, header="true", inferSchema="true", multiLine="true", escape='"')

    # drop the taget column
    df = df.select("customerID", "OnlineSecurity", "OnlineBackup", "DeviceProtection")

    table_name = f"{DA.catalog_name}.{DA.schema_name}.security_features"
    df.write.format("delta").mode("overwrite").saveAsTable(table_name)

# COMMAND ----------

# Initialize DBAcademyHelper
DA = DBAcademyHelper() 
DA.init()