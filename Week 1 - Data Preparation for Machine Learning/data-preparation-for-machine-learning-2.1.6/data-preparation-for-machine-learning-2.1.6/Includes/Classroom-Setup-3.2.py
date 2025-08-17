# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

from pyspark.sql.functions import col, monotonically_increasing_id

# COMMAND ----------

# Create a delta table for diet features
@DBAcademyHelper.add_init
def create_diet_features_table(self):
    
    spark.sql(f"USE CATALOG {DA.catalog_name}")
    spark.sql(f"USE SCHEMA {DA.schema_name}")

    # Set the path of the dataset
    shared_volume_name = 'cdc-diabetes' # From Marketplace
    csv_name = 'diabetes_binary_5050split_BRFSS2015' # CSV file name
    dataset_path = f"{DA.paths.datasets.cdc_diabetes}/{shared_volume_name}/{csv_name}.csv" # Full path


    df = spark.read.csv(dataset_path, header="true", inferSchema="true", multiLine="true", escape='"')
    df = df.select("Fruits", "Veggies", "HvyAlcoholConsump", "Smoker")
    df = df.withColumn("UID", monotonically_increasing_id())

    table_name = f"{DA.catalog_name}.{DA.schema_name}.diet_features"
    df.write.format("delta").mode("overwrite").saveAsTable(table_name)

# COMMAND ----------

# Initialize DBAcademyHelper
DA = DBAcademyHelper() 
DA.init()