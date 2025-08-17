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
def create_dais_table(self):
    source_table_fullname = f"{DA.catalog_name}.{DA.schema_name}.dais_text"
    df = spark.read.load(f"{DA.paths.datasets.dais}/dais_delta")
    df = df.filter(df.Title.isNotNull())
    df = df.withColumn("id", F.monotonically_increasing_id())
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(source_table_fullname)
    print("DAIS dataset is created successfully.")

# COMMAND ----------

# Initialize DBAcademyHelper
DA = DBAcademyHelper() 
DA.init()