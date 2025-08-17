# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

@DBAcademyHelper.add_init
def initialize_uc(self):

    table_name = 'telco_table'
    spark.sql(f"USE CATALOG {DA.catalog_name}")
    spark.sql(f"USE SCHEMA {DA.schema_name}")
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")

    print(f'Using catalog {DA.catalog_name} and schema {DA.schema_name}.')

# COMMAND ----------

# Initialize DBAcademyHelper
DA = DBAcademyHelper() 
DA.init()