# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

@DBAcademyHelper.add_init
def create_features_table(self):
    from pyspark.sql.functions import monotonically_increasing_id, col
    from databricks.feature_engineering import FeatureEngineeringClient
    
    table_name = "bank_loan"
    features_table_name = "bank_loan_features"

    shared_volume_name = "banking"
    csv_name = "loan-clean"

    # Full path to tables in Unity Catalog
    full_table_path = f"{DA.catalog_name}.{DA.schema_name}.{table_name}"
    features_table_path = f"{DA.catalog_name}.{DA.schema_name}.{features_table_name}"

    # Path to CSV file
    #dataset_path = f"{DA.paths.datasets.banking}/{shared_volume_name}/{csv_name}.csv"
    dataset_path = f"/Volumes/dbacademy_banking/v01/banking"
    # Define active catalog and schema
    spark.sql(f"USE CATALOG {DA.catalog_name}")
    spark.sql(f"USE {DA.schema_name}")
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")
    spark.sql(f"DROP TABLE IF EXISTS {features_table_name}")
    
    # Read the CSV file into a Spark DataFrame
    loan_df = (
        spark
        .read
        .format('csv')
        .option('header', True)
        .load(dataset_path)
        )

    # Select columns of interest and replace spaces with underscores
    loan_df_clean = loan_df.selectExpr("ID", "Age", "`ZIP Code` as ZIP_Code", "Family", "CCAvg", "Education", "Mortgage", "`Personal Loan` as Personal_Loan", "`Securities Account` as Securities_Account", "`CD Account` as CD_Account", "Online", "CreditCard")

    # Save df as delta table using Delta API
    loan_df_clean.write.format("delta").mode("overwrite").saveAsTable("bank_loan")

    # Create features table
    load_df_features = loan_df.select("ID", "Experience", "Income")
    fe = FeatureEngineeringClient()
    fe.create_table(
        name = features_table_path,
        primary_keys = ["ID"],
        df = load_df_features,
        description="Bank Loan Feature Table",
        tags={"source": "gold", "format": "delta"}
    )

# COMMAND ----------

# Initialize DBAcademyHelper
DA = DBAcademyHelper() 
DA.init()