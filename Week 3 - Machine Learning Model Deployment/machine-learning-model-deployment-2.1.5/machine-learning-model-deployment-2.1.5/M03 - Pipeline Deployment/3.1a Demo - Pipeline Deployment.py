# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Pipeline Deployment
# MAGIC
# MAGIC In this demo, we will show how to use a model as part of a data pipeline for inference. In the first section of the demo, we will prepare data and perform some basic feature engineering. Then, we will fit and register the model to model registry. Please note that these two steps are already covered in other courses and they are not the main focus of this demo. In the last section, which is the main focus of this demo, we will create a Delta Live Tables (DLT) pipeline and use the registered model as part of the pipeline. 
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to;*
# MAGIC
# MAGIC * Describe steps for deploying a model within a pipeline.
# MAGIC
# MAGIC * Develop a simple Delta Live Tables pipeline that performs batch inference in its final step.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## REQUIRED - SELECT CLASSIC COMPUTE
# MAGIC Before executing cells in this notebook, please select your classic compute cluster in the lab. Be aware that **Serverless** is enabled by default.
# MAGIC
# MAGIC Follow these steps to select the classic compute cluster:
# MAGIC 1. Navigate to the top-right of this notebook and click the drop-down menu to select your cluster. By default, the notebook will use **Serverless**.
# MAGIC
# MAGIC 2. If your cluster is available, select it and continue to the next cell. If the cluster is not shown:
# MAGIC
# MAGIC    - Click **More** in the drop-down.
# MAGIC    
# MAGIC    - In the **Attach to an existing compute resource** window, use the first drop-down to select your unique cluster.
# MAGIC
# MAGIC **NOTE:** If your cluster has terminated, you might need to restart it in order to select it. To do this:
# MAGIC
# MAGIC 1. Right-click on **Compute** in the left navigation pane and select *Open in new tab*.
# MAGIC
# MAGIC 2. Find the triangle icon to the right of your compute cluster name and click it.
# MAGIC
# MAGIC 3. Wait a few minutes for the cluster to start.
# MAGIC
# MAGIC 4. Once the cluster is running, complete the steps above to select your cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **16.3.x-cpu-ml-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-3.1a

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this demo, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"User DB Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Preparation
# MAGIC
# MAGIC For this demonstration, we will utilize a fictional dataset from a Telecom Company, which includes customer information. This dataset encompasses **customer demographics**, including gender, as well as internet subscription details such as subscription plans and payment methods.
# MAGIC
# MAGIC After loading the dataset, we will perform simple **data cleaning and feature selection**. 
# MAGIC
# MAGIC In the final step, we will split the dataset into **features** and **response** sets.

# COMMAND ----------

from pyspark.sql.functions import col

# Load dataset with spark
shared_volume_name = 'telco' # From Marketplace
csv_name = 'telco-customer-churn-missing' # CSV file name
dataset_p_telco = f"{DA.paths.datasets.telco}/{shared_volume_name}/{csv_name}.csv" # Full path

# Dataset specs
primary_key = "customerID"
response = "Churn"
features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"] # Keeping numerical only for simplicity and demo purposes

# Read dataset (and drop nan)
telco_df = spark.read.csv(dataset_p_telco, inferSchema=True, header=True, multiLine=True, escape='"')\
            .withColumn("TotalCharges", col("TotalCharges").cast('double'))\
            .na.drop(how='any')

# Separate features and ground-truth
features_df = telco_df.select(primary_key, *features)
response_df = telco_df.select(primary_key, response)

# Train a sklearn Decision Tree Classification model
# Convert data to pandas dataframes
X_train_pdf = features_df.drop(primary_key).toPandas()
Y_train_pdf = response_df.drop(primary_key).toPandas()

for col in X_train_pdf.select_dtypes("int32"):
    X_train_pdf[col] = X_train_pdf[col].astype("double")

# COMMAND ----------

print(X_train_pdf.info())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Preparation
# MAGIC
# MAGIC **Note:** This section is not the main focus of this course. We are just repeating the model development and registration process here.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Setup Model Registry with UC
# MAGIC
# MAGIC Before we start model deployment, we need to fit and register a model. In this demo, **we will log models to Unity Catalog**, which means first we need to setup the **MLflow Model Registry URI**.

# COMMAND ----------

import mlflow

# Point to UC model registry
mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()


def get_latest_model_version(model_name):
    """Helper function to get latest model version"""
    model_version_infos = client.search_model_versions("name = '%s'" % model_name)
    return max([model_version_info.version for model_version_info in model_version_infos])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit and Register a Model with UC

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier
from mlflow.models import infer_signature

# Use 3-level namespace for model name
model_name = f"{DA.catalog_name}.{DA.schema_name}.ml_model" 

# model to use for classification
clf = DecisionTreeClassifier(max_depth=4, random_state=10)

with mlflow.start_run(run_name="Model-Deployment demo") as mlflow_run:

    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_models=False,
        log_post_training_metrics=True,
        silent=True)
    
    clf.fit(X_train_pdf, Y_train_pdf)

    # Log model and push to registry
    signature = infer_signature(X_train_pdf, Y_train_pdf)
    mlflow.sklearn.log_model(
        clf,
        artifact_path="decision_tree",
        signature=signature,
        registered_model_name=model_name
    )

    # Set model alias
    client.set_registered_model_alias(model_name, "DLT", get_latest_model_version(model_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure Pipeline to Run Batch Inference
# MAGIC
# MAGIC Now that our model is registered and ready, we can move on the most important part; using the model for inference inside a pipeline. 
# MAGIC
# MAGIC **Note: The DLT pipeline is already defined in `3.1b` notebook.**
# MAGIC
# MAGIC **Note: If you want to learn more about DLT please check out Data Engineering with Databricks (Build Data Pipelines with Delta Live Tables).**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Config Variables
# MAGIC
# MAGIC While defining the DLT pipeline, you will need to use the following variables. Run the code block below first. Then, use the output in the next section while creating the pipeline.
# MAGIC

# COMMAND ----------

print(f"mlpipeline.bronze_dataset_path: {dataset_p_telco}")
print(f"mlpipeline.model_name: {model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC ### Create the ETL Pipeline
# MAGIC
# MAGIC To create a pipeline, follow these steps:
# MAGIC
# MAGIC 1. Go to **Jobs & Pipelines** from the left menu.
# MAGIC 1. From the **Create** dropdown, select **ETL Pipeline**.
# MAGIC
# MAGIC **🚨NOTE: Select "Maybe later" to continue creating the pipeline using the current editor and avoid switching to the new ETL Editor.**
# MAGIC
# MAGIC 1. **General** Settings:
# MAGIC     - For **Pipeline name** enter **Demo3.1-Pipeline-[Yourname]**. Make sure the pipeline name is unique.
# MAGIC     - For **Product Edition** choose **Core**. 
# MAGIC     - For **Pipeline mode** choose **Triggered**.
# MAGIC 1. **Source code** Settings:
# MAGIC     - For **Paths** navigate to and select the notebook **3.1b Demo - Inference Pipeline**, in the same folder as this notebook.
# MAGIC 1. **Destination** Settings: 
# MAGIC     - Select **Unity Catalog**.
# MAGIC     - For **Default catalog** and **Default schema**, select the catalog and schema that is created for you in the beginning of this notebook.
# MAGIC 1. **Compute** Settings:
# MAGIC     - Choose **DBAcademy DLT-UC** for the **Cluster policy**.
# MAGIC     - Choose **Fixed size** for **Cluster mode**.
# MAGIC     - Specify **1** for **Workers**.
# MAGIC 1. **Advanced** Settings: 
# MAGIC     - Add these two configurations using the values from previous cell's output. Add the keys and values directly (no quotes needed).
# MAGIC       - `mlpipeline.bronze_dataset_path`
# MAGIC       - `mlpipeline.model_name`
# MAGIC 1. Click on the **Create** button at bottom.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Start the ETL Pipeline
# MAGIC
# MAGIC Initiate the first update to your pipeline by clicking **Start**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demonstration, we walked through the sequential process of training, registering, and deploying a model within a pipeline. Following the standard procedure of fitting and registering the model, we then established a Delta Live Tables pipeline. This pipeline not only ingests data from a source file but also implements necessary data transformations, culminating in the utilization of the registered model as the final step in the pipeline. While your specific project requirements may vary, this example illustrates how to set up and integrate a model for inference within the Delta Live Tables pipeline.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="blank">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy" target="blank">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use" target="blank">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/" target="blank">Support</a>