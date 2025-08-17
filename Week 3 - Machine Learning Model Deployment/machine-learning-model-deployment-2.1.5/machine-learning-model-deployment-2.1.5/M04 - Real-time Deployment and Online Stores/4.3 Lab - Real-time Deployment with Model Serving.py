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
# MAGIC # LAB - Real-time Deployment with Model Serving
# MAGIC
# MAGIC In this lab, you will deploy ML models with Databricks Model Serving **with and without a feature table**. This lab includes **two** sections.
# MAGIC
# MAGIC In the first section, you will deploy a model for real-time inference with Model Serving's **UI**. This section will demonstrate the most basic and simple way of deploying models with Model Serving. 
# MAGIC
# MAGIC For the second section, you will deploy a model with an **online feature table using the API**. 
# MAGIC
# MAGIC For both sections, data preparation, model fitting and model registration are already done for you! You just need to focus on the deployment part.
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC * Simple real-time deployment
# MAGIC   
# MAGIC   - **Task 1:** Serve the model using the UI
# MAGIC   
# MAGIC   - **Task 2:** Query the endpoint
# MAGIC
# MAGIC * Real-time deployment with Online Features
# MAGIC
# MAGIC   - **Task 3**: Create an online feature table
# MAGIC
# MAGIC   - **Task 4:** Deploy a model with the online feature table
# MAGIC
# MAGIC   - **Task 5:** Query the endpoint 
# MAGIC
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
# MAGIC Before starting the lab, run the provided classroom setup scripts. 
# MAGIC
# MAGIC **📌 Note:** In this lab you will be using the Databricks SDK to create Model Serving endpoint. Therefore, you will need to run the next code block to **install `databricks-sdk`**. 
# MAGIC
# MAGIC Before starting the lab, run the provided classroom setup script. This script will define configuration variables necessary for the lab. Execute the following cell:

# COMMAND ----------

# MAGIC %pip install -U -qq databricks-sdk
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-4.3

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this lab, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data and Model Preparation
# MAGIC
# MAGIC Before you start the deployment process, you will need to fit and register a model. In this section, you will load dataset, fit a model and register it with UC.
# MAGIC
# MAGIC **Note:** All necessary code is provided, which means you don't need to complete anything in this section.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Dataset

# COMMAND ----------

from pyspark.sql.functions import col, monotonically_increasing_id

## Set the path of the dataset
shared_volume_name = 'cdc-diabetes' # From Marketplace
csv_name = 'diabetes_binary_5050split_BRFSS2015' # CSV file name
dataset_path = f"{DA.paths.datasets.cdc_diabetes}/{shared_volume_name}/{csv_name}.csv" # Full path


df = spark.read.csv(dataset_path, inferSchema=True, header=True, multiLine=True, escape='"')\
    .na.drop(how='any')

df = df.withColumn("uniqueID", monotonically_increasing_id())   # Add unique_id column

## Dataset specs
primary_key = "uniqueID"
response = "Diabetes_binary"

## Separate features and ground-truth
features_df = df.drop(response)
response_df = df.select(primary_key, response)

## Convert data to pandas dataframes
X_train_pdf = features_df.drop(primary_key).toPandas()
Y_train_pdf = response_df.drop(primary_key).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup Model Registry with UC
# MAGIC
# MAGIC Before we start model deployment, we need to fit and register a model. In this lab, **we will log models to Unity Catalog**, which means first we need to setup the **MLflow Model Registry URI**.

# COMMAND ----------

import mlflow

## Point to UC model registry
mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Helper Class for Model Creation

# COMMAND ----------

import time
import warnings
from mlflow.types.utils import _infer_schema
from mlflow.models import infer_signature
from sklearn.tree import DecisionTreeClassifier
from databricks.feature_engineering import FeatureEngineeringClient

model_name = f"{DA.catalog_name}.{DA.schema_name}.ml_diabetes_model" ## Use 3-level namespace

def get_latest_model_version(model_name):
    """Helper function to get the latest model version as a string"""
    model_version_infos = client.search_model_versions("name = '%s'" % model_name)
    model_version_list = [model_version_info.version for model_version_info in model_version_infos]
    ## Convert to integers for correct numeric comparison
    model_version_int_list = list(map(int, model_version_list))
    ## Find the maximum and convert back to a string
    return str(max(model_version_int_list))

def fit_and_register_model(X, Y, model_name_=model_name, random_state_=42, model_alias=None, log_with_fs=False, training_set_spec_=None):
    """Helper function to train and register a decision tree model"""

    clf = DecisionTreeClassifier(random_state=random_state_)
    with mlflow.start_run(run_name="LAB4-Real-Time-Deployment") as mlflow_run:

        ## Enable automatic logging of input samples, metrics, parameters, and models
        mlflow.sklearn.autolog(
            log_input_examples=True,
            log_models=False,
            log_post_training_metrics=True,
            silent=True)
        
        clf.fit(X, Y)

        ## Log model and push to registry
        if log_with_fs:
            # Infer output schema
            try:
                output_schema = _infer_schema(Y)
            except Exception as e:
                warnings.warn(f"Could not infer model output schema: {e}")
                output_schema = None
            
            ## Log using feature engineering client and push to registry
            fe = FeatureEngineeringClient()
            fe.log_model(
                model = clf,
                artifact_path = "decision_tree",
                flavor = mlflow.sklearn,
                training_set = training_set_spec_,
                output_schema = output_schema,
                registered_model_name = model_name_
            )
        
        else:
            signature = infer_signature(X, Y)
            example = X[:3]
            mlflow.sklearn.log_model(
                clf,
                artifact_path = "decision_tree",
                signature = signature,
                input_example = example,
                registered_model_name = model_name_
            )

        ## Set model alias
        if model_alias:
            time.sleep(10) ## Wait 10secs for model version to be created
            client.set_registered_model_alias(model_name_, model_alias, get_latest_model_version(model_name_))

    return clf

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit and Register the Model
# MAGIC
# MAGIC Before we start model deployment process, we will **fit and register a model**. The model's alias will be set to `Production` and it will be served with Databricks Model Serving in the next step.

# COMMAND ----------

model = fit_and_register_model(X_train_pdf, Y_train_pdf, model_name, 42, "Production")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simple Real-time Model Deployment
# MAGIC
# MAGIC Now that the model is registered and ready for deployment, the next step is to create a serving endpoint with Model Serving and serve the model.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1: Serve the Model Using the UI
# MAGIC
# MAGIC Serve the **"Production"** model that we registered in the previous section using the following endpoint configuration.
# MAGIC
# MAGIC **Configuration:**
# MAGIC
# MAGIC * Name: `la4-1-diabetes-model`
# MAGIC
# MAGIC * Compute Size: `small` (CPU)
# MAGIC
# MAGIC * Autoscaling: `Scale to zero`
# MAGIC
# MAGIC * Tags: Define tags that might be meaningful for this deployment
# MAGIC
# MAGIC
# MAGIC **💡 Note:** Endpoint creation will take sometime. Therefore, you can work on the next section  while the endpoint is created for you.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2: Query the Endpoint 
# MAGIC
# MAGIC Test the model deployment using the **Query endpoint** feature in browsers. Use the provided **Example request** payload to use the model for inference.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Real-time Model Deployment with Online Store
# MAGIC
# MAGIC In this section you will deploy a model with a feature table using Databricks' Online Tables. Also, instead of using the UI for creating and configuring the serving endpoint, this time you will need to use the API. 
# MAGIC
# MAGIC First you will create a feature table. Then, you will create an online table that based on the feature table. As we will compute a feature "on-demand", you will create a function to compute the new feature on-demand. Finally, you will fit the model and serve it with Model Serving.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3: Create Feature Table
# MAGIC
# MAGIC Let's create a feature table to store the features that will be use for training the model. 
# MAGIC
# MAGIC For this task you will need setup the feature as follows;
# MAGIC - The feature table will use all fields
# MAGIC - Setup the primary key
# MAGIC - Define description for the table

# COMMAND ----------

from databricks.feature_engineering import FeatureLookup, FeatureEngineeringClient

feature_table_name = f"{DA.catalog_name}.{DA.schema_name}.diabetes_features"
fe = FeatureEngineeringClient()

## Create feature table
fe.create_table(
    name=feature_table_name,
    df=features_df,
    primary_keys=[primary_key],
    description="Diabetes features table",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 4: Create a Function
# MAGIC
# MAGIC Instead of directly using **Education** and **Income**, a new field called *Education-Adjusted Income Index (EAI)* will be calculated and used in this model.
# MAGIC
# MAGIC This field is computed using the formula:  
# MAGIC **`Education-Adjusted Income = Income × Education Weight`**
# MAGIC
# MAGIC *Note:* This field will likely be highly correlated with both income and education. In real-world applications, such correlations should be carefully addressed. However, the purpose here is to demonstrate how on-demand feature computation works.
# MAGIC
# MAGIC The function should be structured as follows, using the variable names as defined below:  
# MAGIC -  **Function name**: `eai_function` 
# MAGIC - **Input:** `Income`, `Education`  
# MAGIC - **Output:** `eai`
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION eai_function (Income DOUBLE, Education DOUBLE)
RETURNS DOUBLE
LANGUAGE PYTHON AS
$$
eai = Income * Education
return eai
$$""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 5: Create an Online Table
# MAGIC
# MAGIC In this task, you will need to integrate the feature table for inference. For real-time inference, Model Serving will need to access features in real-time. 
# MAGIC
# MAGIC **Create an online feature table using following configurations:**
# MAGIC
# MAGIC * Enable CDF for the source table
# MAGIC
# MAGIC * Online table name: `diabetes_online_feature_table`
# MAGIC
# MAGIC * Sync mode: `Triggered`

# COMMAND ----------

from pprint import pprint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy

online_table_name=f"{DA.catalog_name}.{DA.schema_name}.diabetes_online_feature_table"

workspace = WorkspaceClient()

## Enable CDF for the table
spark.sql(f"""ALTER TABLE {feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)""")

## Create an online table
spec = OnlineTableSpec(
  primary_key_columns = [primary_key],
  source_table_full_name = feature_table_name,
  run_triggered = OnlineTableSpecTriggeredSchedulingPolicy.from_dict({'triggered': 'true'}),
  perform_full_copy = True
)

online_table = OnlineTable(
    name=online_table_name,
    spec=spec)

try:
  online_table_pipeline = workspace.online_tables.create_and_wait(table=online_table)
except Exception as e:
  if "already exists" in str(e):
    pass
  else:
    raise e

pprint(workspace.online_tables.get(online_table_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 6: Define Features
# MAGIC
# MAGIC Now that you have both an online feature table and a function, you will combine these together to be used for by the model.

# COMMAND ----------

from databricks.feature_engineering import FeatureFunction
## Define combined feature lookup and feature function
features=[
    FeatureLookup(
        table_name=feature_table_name, 
        lookup_key=primary_key),
    FeatureFunction(
        udf_name="eai_function",
        output_name="eai",
        input_bindings={
            "Income": "Income",
            "Education": "Education"
        },
    )
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 7: Create Training Set and Fit the Model
# MAGIC
# MAGIC Now that all feature configuration is set and ready, create training set and fit the model.

# COMMAND ----------

## Define training set
training_set_spec = fe.create_training_set(
    df=response_df,
    label=response,
    feature_lookups=features,
    exclude_columns=[primary_key]
)

## Load training dataframe based on defined feature-lookup specification
training_df = training_set_spec.load_df()

## Convert data to pandas dataframes
X_train_pdf2 = training_df.drop(primary_key, response).toPandas()
Y_train_pdf2 = training_df.select(response).toPandas()

## Fit and register the mode
model_name_2 = f"{DA.catalog_name}.{DA.schema_name}.ml_diabetes_model_fe"
model_fe = fit_and_register_model(X_train_pdf2, Y_train_pdf2, model_name_2, 20, log_with_fs=True, training_set_spec_=training_set_spec)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 8: Deploy the Model with Online Store
# MAGIC
# MAGIC Create an endpoint with following configuration;
# MAGIC
# MAGIC * Autoscaling: `Scale-to-zero`
# MAGIC
# MAGIC * Compute size: `Small`
# MAGIC
# MAGIC **💡 Note:** Endpoint creation will take sometime. Be patient while the endpoint is created.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, EndpointTag

## Create/Update endpoint and deploy model+version
w = WorkspaceClient()

## Get model version that will be served
fs_model_version = get_latest_model_version(model_name_2)

## Endpoint configuration
fs_endpoint_config_dict = {
   "served_models": [
       {
          "model_name": model_name_2,
          "model_version": fs_model_version,
          "scale_to_zero_enabled": True,
          "workload_size": "Small"
       }
   ]
}
fs_endpoint_config = EndpointCoreConfigInput.from_dict(fs_endpoint_config_dict)  


fs_endpoint_name = f"ML_AS_03_Lab4_FS_{DA.unique_name('_')}"
try:
  w.serving_endpoints.create_and_wait(
    name=fs_endpoint_name,
    config=fs_endpoint_config,
    tags=[EndpointTag.from_dict({"key": "db_academy", "value": "lab4_serve_fs_model"})]
  )
  
  print(f"Creating endpoint {fs_endpoint_name} with models {model_name} versions {fs_model_version}")

except Exception as e:
  if "already exists" in e.args[0]:
    print(f"Endpoint with name {fs_endpoint_name} already exists")

  else:
    raise(e)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 9: Query the Endpoint
# MAGIC
# MAGIC After the endpoint is created, it is time to test it. Use the following hard-coded test-sample to query the endpoint using the API.

# COMMAND ----------

## Hard-coded test-sample. Feel free to change the ids
dataframe_records_lookups_only = [
    {"uniqueID": "123"},
    {"uniqueID": "45678"}
]

# COMMAND ----------

## Query the serving endpoint with test-sample
query_response = w.serving_endpoints.query(
    name=fs_endpoint_name,
    dataframe_records=dataframe_records_lookups_only)
print(f"FS Inference results: {query_response.predictions}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC Great job for completing this lab! In this lab, you completed two main tasks: deploying a model with Model Serving using both with and without feature store tables. In the first section of the lab, the main task was to deploy a model simply using the UI. The second section focused on registering a model with a feature table, creating an online feature table from an existing table, and serving a model with an online feature store. Additionally, for each of these methods, there was an endpoint query task to test the endpoint.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="blank">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy" target="blank">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use" target="blank">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/" target="blank">Support</a>