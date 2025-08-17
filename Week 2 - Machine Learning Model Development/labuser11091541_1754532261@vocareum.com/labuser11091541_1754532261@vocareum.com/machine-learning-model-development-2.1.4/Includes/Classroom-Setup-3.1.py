# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

@DBAcademyHelper.add_init
def create_features_table(self):
    from databricks.feature_engineering import FeatureEngineeringClient
    from pyspark.sql.functions import monotonically_increasing_id, col

    table_name = 'customer_churn'
    features_table_name = "customer_churn_features"

    shared_volume_name = "telco"
    csv_name = "telco-customer-churn"

    # Full path to tables in Unity Catalog
    full_table_path = f"{DA.catalog_name}.{DA.schema_name}.{table_name}"
    features_table_path = f"{DA.catalog_name}.{DA.schema_name}.{features_table_name}"

    # Path to CSV file
    dataset_path = f"{DA.paths.datasets.telco}/{shared_volume_name}/{csv_name}.csv"

    # Define active catalog and schema
    spark.sql(f"USE CATALOG {DA.catalog_name}")
    spark.sql(f"USE {DA.schema_name}")
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")
    spark.sql(f"DROP TABLE IF EXISTS {features_table_name}")
    
    # Read the CSV file into a Spark DataFrame
    telco_df = (
        spark
        .read
        .format('csv')
        .option('header', True)
        .load(dataset_path)
        )

    # basic clean-up
    telco_df = telco_df.withColumn("CustomerID",  monotonically_increasing_id())
    telco_df = telco_df.withColumn("Gender", col("gender"))
    telco_df = telco_df.withColumn("Tenure", col("tenure").cast("double"))
    telco_df = telco_df.withColumn("TotalCharges", col("TotalCharges").cast("double"))
    telco_df = telco_df.withColumn("AverageMonthlyCharges", col("TotalCharges")/col("Tenure"))

    # Select columns of interest
    telco_df_clean = telco_df.select("CustomerID", "Gender", "SeniorCitizen", "Partner", "InternetService", "Contract", "PaperlessBilling", "PaymentMethod", "Churn")

    # save df as delta table
    telco_df_clean.write.format("delta").option("overwriteSchema", 'true').mode("overwrite").saveAsTable("customer_churn")

    # Create features table
    df_features = telco_df.select("CustomerID", "AverageMonthlyCharges")
    fe = FeatureEngineeringClient()
    fe.create_table(
        name=features_table_path,
        primary_keys=["CustomerID"],
        df=df_features,
        description="Customer Churn Feature Table",
        tags={"source": "gold", "format": "delta"}
    )

# COMMAND ----------

# Initialize DBAcademyHelper
DA = DBAcademyHelper() 
DA.init()