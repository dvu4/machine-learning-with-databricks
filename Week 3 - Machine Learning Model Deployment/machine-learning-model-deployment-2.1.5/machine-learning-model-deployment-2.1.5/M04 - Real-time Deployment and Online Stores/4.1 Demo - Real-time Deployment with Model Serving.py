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
# MAGIC # Real-time Deployment with Model Serving
# MAGIC
# MAGIC In this demo, we will focus on real-time deployment of machine learning models. Databricks' Model Serving is an easy-to-use serverless infrastructure for serving the models in real-time that supports both online and offline feature tables as well as automatic feature lookups for online tables with no additional endpoint configuration.  
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to;*
# MAGIC
# MAGIC - Understand the differences between **offline** and **online** feature tables for Databricks Model Serving.
# MAGIC - Understand how to serve multiple versions of a model simultaneously and set up **A/B testing** for real-time inferencing.
# MAGIC - Utilize **feature lookups** and a **feature function** for online tables for real-time inference.
# MAGIC
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
# MAGIC
# MAGIC * Online Tables must be enabled for the workspace.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-4.1

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
# MAGIC ## Offline vs. Online Feature Tables For Real-Time Inferencing
# MAGIC
# MAGIC Let's take a moment to discuss the importance of feature tables with real-time model serving. 
# MAGIC
# MAGIC We make the distinction to demonstrate real-time model serving _with_ and _without_ utilizing feature lookups, since the setup for utilizing offline and online tables are handled differently with Model Serving on Databricks. When using real-time model serving with Databricks, you can use **offline** tables _without_ utilizing feature lookups or **online** tables _with_ feature lookups (in which case Databricks provides automatic feature lookup). To utilize **offline** tables with feature lookups, there is batch inferencing via the `score_batch` method from the Databricks SDK.
# MAGIC
# MAGIC
# MAGIC >Fundamentally, ****a feature table is a materialized Delta table with a primary key**** that we want to use for model training. Feature lookups must be configured prior to training your model. For further reading and references of offline vs online feature tables, see the Appendix at the end of this demo. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Real-time Deployment With Offline Feature Tables
# MAGIC Here we consider a scenario where you have already gone through the development process (data preparation, and model development) and you're ready to deploy a model with offline features. We will first look at deploying two models that were created as a part of the classroom setup - a champion model and a challenger model with aliases `champion` and `challenger`, respectively. 
# MAGIC
# MAGIC We will serve our two models using a 50/50 traffic split for A/B Testing. First, Let's read in our data and explore its lineage. 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Inspect Offline The Feature Table and Model Versions
# MAGIC
# MAGIC For this demonstration, we will use a fictional dataset from a Telecom Company, which includes customer information. This dataset encompasses **customer demographics**, including internet subscription details such as subscription plans, monthly charges and payment methods. 
# MAGIC
# MAGIC As a part of the classroom setup for this course, a feature table was created called **features** that **did _not_ include feature lookups.** This is the table we are reading in during the next step.
# MAGIC
# MAGIC #### Lineage Inspection
# MAGIC - Navigate to the catalog and schema used with this Vocareum environment (see the output from the previous cell).
# MAGIC - Find the table called `features` and model called `ml_model`. 
# MAGIC   - Click on Lineage. 
# MAGIC   - Click on **See lineage graph** and inspect it. This will show the footprint of how the catalog assets were made.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Read in Features and Response Variable from Feature Store
# MAGIC
# MAGIC Here we will read in our dataset and split between features and response variables. We will show how this can be performed with the Databricks SDK using the Feature Engineering Client.
# MAGIC
# MAGIC > #### What's the difference between `fe.read_table()` and `read.spark.table()`?
# MAGIC Essentially, we use `fe.read_table()` whenever we are specifically working with feature tables stored within Feature Store and `spark.read.table()` for general-purpose reading. Note that `fe.read_table()` is part of the Databricks Feature Engineering API and integrates well with other Feature Store APIs like logging models (see ****Part 2: Real-Time Deployment with Online Feature Tables****). On the other hand, `spark.read.table()` is a broader Spark SQL method for reading data from any table within the Spark session.

# COMMAND ----------

# DBTITLE 1,Feature Engineering Client Setup and Data Preparation
from databricks.feature_engineering import FeatureEngineeringClient

# Initialize Feature Engineering Client
fe = FeatureEngineeringClient()

# Define primary key 
primary_key = "customerID"

# Read in feature table
feature_table_name = f"{DA.catalog_name}.{DA.schema_name}.features"
X_train_df = fe.read_table(name=feature_table_name)
X_train_pdf = X_train_df.drop(primary_key).toPandas()

# Read in response table 
response_table_name = f"{DA.catalog_name}.{DA.schema_name}.response"
Y_train_df = spark.read.table(response_table_name)
Y_train_pdf = Y_train_df.drop(primary_key).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Real-time A/B Testing with Model Serving
# MAGIC
# MAGIC Let's serve the two models we logged in the previous step using Model Serving. Model Serving supports endpoint management via the UI and the API. 
# MAGIC
# MAGIC Below you will find instructions for using the UI and it is simpler method compared to the API. **In this demo, we will use the API to configure and create the endpoint**.
# MAGIC
# MAGIC **Both the UI and the API support querying created endpoints in real-time**. We will use the API to query the endpoint using a test-set.
# MAGIC
# MAGIC
# MAGIC > #### What is A/B Testing? 
# MAGIC > A/B testing is a method to compare two versions of a model or system by splitting user traffic and measuring performance metrics to determine which version delivers better results.

# COMMAND ----------

# DBTITLE 1,Generate Unique Endpoint Name and Print It
endpoint_name = f"ML_AS_03_Demo4_{DA.unique_name('_')}"
print(f"Endpoint name: {endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 1: Serve model(s) using UI
# MAGIC
# MAGIC After registering the (new version(s) of the) model to the model registry. To provision a serving endpoint via UI, follow the steps below.
# MAGIC
# MAGIC 1. In the left sidebar, click **Serving**.
# MAGIC
# MAGIC 2. To create a new serving endpoint, click **Create serving endpoint**.   
# MAGIC   
# MAGIC     a. In the **Name** field, enter the name printed above.  
# MAGIC   
# MAGIC     b. Click in the Entity field. A dialog appears. Go to **My models**, and then select the **'ml_model'** from the drop-down menus. 
# MAGIC
# MAGIC     c. Click **Confirm**.
# MAGIC   
# MAGIC     d. In the **Version** drop-down menu, select the **version 1**.    
# MAGIC   
# MAGIC     e. In the **Compute Scale-out** drop-down, select Small, Medium, or Large. If you want to use GPU serving, select a GPU type from the **Compute type** drop-down menu.
# MAGIC   
# MAGIC     f. *[OPTIONAL]* to deploy another model (e.g. for A/B testing):
# MAGIC     - Click on **+Add served entity**.
# MAGIC     - Enter the above mentioned details as above, but use **version 2**.
# MAGIC     - Set the traffic split to 50% for each model.
# MAGIC   
# MAGIC     g. Click **Create**. The endpoint page opens and the endpoint creation process starts.   
# MAGIC   
# MAGIC See the Databricks documentation for details ([AWS](https://docs.databricks.com/machine-learning/model-serving/create-manage-serving-endpoints.html#ui-workflow)|[Azure](https://learn.microsoft.com/azure/databricks/machine-learning/model-serving/create-manage-serving-endpoints#--ui-workflow)).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 2: Serve Model(s) Using the Databricks Python SDK
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### Get Models to Serve
# MAGIC
# MAGIC In order to serve the model, we will initialize the MLflow client with `MLflowClient` and the workspace client with `WorkspaceClient`. We will configure the MLflow client to point to Unity Catalog instead of the Workspace with `set_registry_uri("databricks-uc")`. The workspace client will be used to create the model serving endpoint.

# COMMAND ----------

# DBTITLE 1,Run to Initialize MLflow client and workspace client
from mlflow.tracking import MlflowClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointTag

# Point to UC model registry
mlflow.set_registry_uri("databricks-uc")
# Initialize MLflow client
client = mlflow.MlflowClient()
# Initialize workspace client
w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC Define variables that will be used for configuring the endpoint like `model_name`. The output from running the next cell will show version 1 of our model registered as the champion model and version 2 as being the challenger.

# COMMAND ----------

# DBTITLE 1,Retrieve and Display Model Versions
# Define model name
model_name = f"dbacademy.{DA.schema_name}.ml_model"
# Parse model name from UC namespace
served_model_name =  model_name.split('.')[-1]
# Define the endpoint name
endpoint_name = f"ML_AS_03_Demo4_{DA.unique_name('_')}"

# Get version of our model registered to UC as a part of the classroom setup
model_version_champion = client.get_model_version_by_alias(name=model_name, alias="Champion").version # Get champion version
model_version_challenger = client.get_model_version_by_alias(name=model_name, alias="Challenger").version # Get challenger version


print(f"Model version Champion: {model_version_champion}")
print(f"Model version Challenger: {model_version_challenger}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Configure
# MAGIC
# MAGIC Define our model serving endpoint with `endpoint_config`. The configuration below shows two versions of the same being deployed (`model_version_champion` and `model_version_challenger`) along with how to configure traffic during inferencing.

# COMMAND ----------

# DBTITLE 1,Skip this cell if you created the endpoint using the UI.
from databricks.sdk.service.serving import EndpointCoreConfigInput

endpoint_config_dict = {
    "served_models": [
        {
            "model_name": model_name,
            "model_version": model_version_champion,
            "scale_to_zero_enabled": True,
            "workload_size": "Small"
        },
        {
            "model_name": model_name,
            "model_version": model_version_challenger,
            "scale_to_zero_enabled": True,
            "workload_size": "Small"
        },
    ],
    "traffic_config": {
        "routes": [
            {"served_model_name": f"{served_model_name}-{model_version_champion}", "traffic_percentage": 50},
            {"served_model_name": f"{served_model_name}-{model_version_challenger}", "traffic_percentage": 50},
        ]
    },
    "auto_capture_config":{
        "catalog_name": DA.catalog_name,
        "schema_name": DA.schema_name,
        "table_name_prefix": "db_academy"
    }
}


endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Serve the endpoint
# MAGIC Use the configuration just created to serve the model.
# MAGIC > The time to create a model serving endpoint is roughly 10 minutes on average.

# COMMAND ----------

# DBTITLE 1,Skip this cell if you created the endpoint using the UI.
try:
  w.serving_endpoints.create_and_wait(
    name=endpoint_name,
    config=endpoint_config,
    tags=[EndpointTag.from_dict({"key": "db_academy", "value": "serve_fs_model_example"})]
  )
  print(f"Creating endpoint {endpoint_name} with models {model_name} versions {model_version_champion} & {model_version_challenger}")

except Exception as e:
  if "already exists" in e.args[0]:
    print(f"Endpoint with name {endpoint_name} already exists")

  else:
    raise(e)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Verify Endpoint Creation
# MAGIC
# MAGIC Let's verify that the endpoint is created and ready to be used for inference using the `assert` command, which is used to check whether a given condition is true.

# COMMAND ----------

# DBTITLE 1,Run this to test if endpoint is ready
endpoint = w.serving_endpoints.wait_get_serving_endpoint_not_updating(endpoint_name)

assert endpoint.state.config_update.value == "NOT_UPDATING" and endpoint.state.ready.value == "READY" , "Endpoint not ready or failed"

# COMMAND ----------

# MAGIC %md
# MAGIC #### Query the Endpoint and Visualize
# MAGIC
# MAGIC Here we will use the training dataset to query our endpoint.
# MAGIC
# MAGIC 1. Define the dataset to sample from.
# MAGIC 1. Query by batch to highlight model-split traffic.

# COMMAND ----------

# DBTITLE 1,Sample 1,000 Records from X_train_pdf
dataframe_records = X_train_pdf.iloc[:1000].to_dict(orient='records') #1k sample records

# COMMAND ----------

# MAGIC %md
# MAGIC Here we will query in batches so we can see the traffic split per 100 rows (there are around 2000 rows in this dataset)
# MAGIC
# MAGIC To help visualize the A/B testing output, create a visual using the UI (you only need to do this once;  rerunning the cell will update the visualization). 
# MAGIC 1. After running the next cell, select the + sign on the second table and select **Visualization**. 
# MAGIC 1. The default visual should represent the Yes/No split per model.
# MAGIC
# MAGIC > Since the dataset we're working with is not very large, you might have to run the cell a few times to get a fairly close 50/50 split.

# COMMAND ----------

# DBTITLE 1,Batch Processing and Aggregation of Model Predictions
import pandas as pd

print("Inference results:")

batch_size = 100  # Number of records per batch
num_batches = (len(dataframe_records) + batch_size - 1) // batch_size  # Total number of batches

all_predictions = []
all_models = []

# Process data in batches
for i in range(num_batches):
    batch_records = dataframe_records[i * batch_size:(i + 1) * batch_size]  # Slice batch

    # Query the model serving endpoint
    query_response = w.serving_endpoints.query(name=endpoint_name, dataframe_records=batch_records)

    # Collect predictions and model served details
    all_predictions.extend(query_response.predictions)
    all_models.extend([query_response.served_model_name] * len(query_response.predictions))  # Duplicate model name per prediction

# Convert to DataFrame
results_df = pd.DataFrame({
    "prediction": all_predictions,
    "model_served": all_models
})

# Count occurrences of predictions
count_results = results_df['prediction'].value_counts().reset_index()
count_results.columns = ['prediction', 'count']

# Display aggregated count of predictions
display(count_results)

# Aggregate count of predictions per model
model_count_results = results_df.groupby(["model_served", "prediction"]).size().reset_index(name="count")

# Display results grouped by model and prediction type
display(model_count_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Real-time Deployment with Online Feature Tables
# MAGIC
# MAGIC In the previous section we deployed a model that utilized an offline feature table without utilizing feature lookups. In this section we will build a model that utilizes feature lookups with an online table and serve this model. Here are the steps we will take: 
# MAGIC 1. Create a feature function that computes the average monthly usage charges per customer.
# MAGIC 1. Bundle the feature lookups and feature function into one feature-defining object called `features`
# MAGIC 1. Use the Databricks SDK to create the online feature table using the same feature table from part 1. 
# MAGIC 1. Train an ML model using `features`. By creating a model using feature lookups, we will enable automatic feature lookups when deploying the model to a model serving endpoint. This requires no additional configuration with the online feature table. 
# MAGIC 1. Create a model serving endpoint.
# MAGIC 1. Query the model serving endpoint.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1: Create a Feature Function
# MAGIC Here we will create a feature function that uses a Python UDF to create on-demand features.
# MAGIC
# MAGIC #### On-Demand Features
# MAGIC
# MAGIC “On-demand” refers to features whose values are not known ahead of time, but are **_calculated at the time of inference_**. In this demo, we will calculate the **average monthly charges** on the fly. This is done by defining **a UDF** with SQL and registering it to Unity Catalog. **The function will be registered** with the name `monthly_charges_avrg` using the syntax `CREATE OR REPLACE FUNCTION`.

# COMMAND ----------

# DBTITLE 1,Function to Calculate Average Monthly Charges
# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION monthly_charges_avrg (TotalCharges DOUBLE, tenure DOUBLE)
# MAGIC RETURNS DOUBLE
# MAGIC LANGUAGE PYTHON AS
# MAGIC $$
# MAGIC avrg = TotalCharges / tenure
# MAGIC return avrg
# MAGIC $$

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Define Combined Features
# MAGIC
# MAGIC Now that we have both an **online features table** and **on-demand features** created, we can combine these together to be passed to the model training. We combine these two Unity Catalog assets and store them as a single object called `features`. Then use the Databricks SDK to create a `FeatureSpec` to bundle features into single feature-defining object. 
# MAGIC
# MAGIC > Feature lookups and feature functions must be created prior to training the model.

# COMMAND ----------

# DBTITLE 1,Generate Full Online Feature Name
features_for_online_name = f"{DA.catalog_name}.{DA.schema_name}.features"

# COMMAND ----------

# DBTITLE 1,Feature Lookup and Function for Monthly Charges
from databricks.feature_engineering import FeatureLookup, FeatureFunction

fe = FeatureEngineeringClient()

features=[
  FeatureLookup(
    table_name=features_for_online_name,
    lookup_key=primary_key
  ),
  FeatureFunction(
    udf_name="monthly_charges_avrg", 
    output_name="m_charges_avrg",
    input_bindings={
      "TotalCharges": "TotalCharges", 
      "tenure": "tenure"
    },
  ),
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Create an Online Table
# MAGIC
# MAGIC In this section, we will create an online table to serve feature table for real-time inference. When using Model Serving to serve a model that was built using features from Databricks, the model automatically looks up and transforms features for inference requests
# MAGIC
# MAGIC > Databricks Online Tables can be created and managed via the UI and the SDK. While we provided instructions for both of these methods, you can pick one option for creating the table.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC #### OPTION 1: Create Online Table via the UI
# MAGIC
# MAGIC You create an online table from the Catalog Explorer. The steps are described below. For more details, see the Databricks documentation ([AWS](https://docs.databricks.com/en/machine-learning/feature-store/online-tables.html#create)|[Azure](https://learn.microsoft.com/azure/databricks/machine-learning/feature-store/online-tables#create)). For information about required permissions, see Permissions ([AWS](https://docs.databricks.com/en/machine-learning/feature-store/online-tables.html#user-permissions)|[Azure](https://learn.microsoft.com/azure/databricks/machine-learning/feature-store/online-tables#user-permissions)).
# MAGIC
# MAGIC
# MAGIC In **Catalog Explorer**, navigate to the source table that you want to sync to an online table. 
# MAGIC
# MAGIC Click on the **Create** button and, select **Online table**.
# MAGIC
# MAGIC * Use the selectors in the dialog to configure the online table.
# MAGIC   
# MAGIC   * `Name`: Name to use for the online table in Unity Catalog.
# MAGIC   
# MAGIC   * `Primary Key`: Column(s) in the source table to use as primary key(s) in the online table.
# MAGIC   
# MAGIC   * Timeseries Key: (Optional). Column in the source table to use as timeseries key. When specified, the online table includes only the row with the latest timeseries key value for each primary key.
# MAGIC   
# MAGIC   * `Sync mode`:  Select **`Snapshot`** for Sync mode. Please refer to the documentation for more details about available options.
# MAGIC
# MAGIC   * When you are done, click Confirm. The online table page appears.
# MAGIC
# MAGIC The new online table is created under the catalog, schema, and name specified in the creation dialog. In Catalog Explorer, the online table is indicated by online table icon.

# COMMAND ----------

# MAGIC %md
# MAGIC #### OPTION 2: Use the Databricks SDK 
# MAGIC
# MAGIC The first option for creating an online table is using the UI. The alternative is using Databricks' [python-sdk](https://databricks-sdk-py.readthedocs.io/en/latest/workspace/catalog/online_tables.html). Let's  first define the table specifications, then create the table.
# MAGIC
# MAGIC **🚨 Note:** The workspace must be enabled for using the SDK for creating and managing online tables. You can run following code blocks if your workspace is enabled for this feature.
# MAGIC
# MAGIC > The following code alters your existing feature table using change data feed (CDF). Essentially, this allows tracking of row-level changes between versions of our feature table (any Delta table in general).

# COMMAND ----------

# MAGIC %md
# MAGIC Define the name for our online table.

# COMMAND ----------

# DBTITLE 1,Initialize Online Table Specification and Naming
from databricks.sdk.service.catalog import OnlineTableSpec, OnlineTable, OnlineTableSpecTriggeredSchedulingPolicy

online_table_name=f"{DA.catalog_name}.{DA.schema_name}.online_features"

# COMMAND ----------

# MAGIC %md
# MAGIC **Note🚨**: If the online table table already exists, drop it and enable Change Data Feed (CDF).
# MAGIC <br> Do not run this step if you are using Option 1 to create the online table through the UI.Please continue with Step 4.

# COMMAND ----------

# DBTITLE 1,Skip this cell if you created the online table using the UI.
try: 
  # Drop the online table if it already exists
  w.online_tables.delete(online_table_name)
except:
  pass

# Enable CDF for the table
spark.sql(f"""ALTER TABLE {features_for_online_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)""")

# COMMAND ----------

# MAGIC %md
# MAGIC Configure online table initialization

# COMMAND ----------

# DBTITLE 1,Skip this cell if you created the online table using the UI.
# Create an online table
spec = OnlineTableSpec(
  primary_key_columns = [primary_key],
  source_table_full_name = features_for_online_name,
  run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({'triggered': 'true'}),
  perform_full_copy=True)

online_table = OnlineTable(
  name = online_table_name,
  spec = spec
)

# COMMAND ----------

# MAGIC %md
# MAGIC Create an online table based off `features`.
# MAGIC > ****Does this mean my original model versions are now using feature lookups?**** No. Because feature lookups were _not_ configured during model training, the model serving endpoint will not "know" to perform automatic feature lookup. This step simply syncs the feature table to an online table.

# COMMAND ----------

# DBTITLE 1,Skip this cell if you created the online table using the UI.
# Initialize workspace client
w = WorkspaceClient()
try:
  online_table_pipeline = w.online_tables.create_and_wait(table=online_table)
except Exception as e:
  if "already exists" in str(e):
    pass
  else:
    raise e

# COMMAND ----------

# DBTITLE 1,Retrieve and Display Specific Online Table
print(w.online_tables.get(online_table_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4: Fit and Log the Model with Online Feature Table
# MAGIC Next, we will use the feature engineering client from the Databricks SDK to create our training set that includes the `feature_lookups` parameter - which is our bundled `features` object from the previous cell. 
# MAGIC
# MAGIC The function `fit_and_register_model` is used in the cell below. This function is created as a part of the classroom setup, but we provide the code here for completeness. 
# MAGIC
# MAGIC ```
# MAGIC def fit_and_register_model(
# MAGIC     feature_df, 
# MAGIC     response_df, 
# MAGIC     model_name_, 
# MAGIC     random_state_, 
# MAGIC     model_alias=None, 
# MAGIC     training_set_spec_=None,
# MAGIC     is_online=False
# MAGIC     ):
# MAGIC     """Train and register a Decision Tree model."""
# MAGIC     clf = DecisionTreeClassifier(random_state=random_state_)
# MAGIC     if is_online:
# MAGIC
# MAGIC         X = feature_df # pyspark dataframe
# MAGIC         y = response_df # pyspark dataframe
# MAGIC
# MAGIC         
# MAGIC     else: 
# MAGIC         feature_pdf = feature_df.df.toPandas() #instance of an mlflow.data.Dataset
# MAGIC         response_pdf = response_df.df.toPandas() #instance of an mlflow.data.Dataset
# MAGIC
# MAGIC         dataset = feature_pdf.merge(response_pdf, on="customerID", how="inner")
# MAGIC         # Prepare X and y
# MAGIC         X = dataset.drop(columns=["customerID", "Churn"])  # Drop unnecessary columns
# MAGIC         y = dataset["Churn"]
# MAGIC
# MAGIC     with mlflow.start_run(run_name=f"Train_DecisionTree_{random_state_}"):
# MAGIC         mlflow.sklearn.autolog(
# MAGIC             log_input_examples=True,
# MAGIC             log_models=False,
# MAGIC             log_post_training_metrics=True,
# MAGIC             silent=True
# MAGIC         )
# MAGIC
# MAGIC         clf.fit(X, y) # Fit the model
# MAGIC         
# MAGIC
# MAGIC         # Log model
# MAGIC         if is_online:
# MAGIC             try:
# MAGIC                 output_schema = _infer_schema(y)
# MAGIC             except Exception as e:
# MAGIC                 warnings.warn(f"Could not infer model output schema: {e}")
# MAGIC                 output_schema = None
# MAGIC             
# MAGIC             fe = FeatureEngineeringClient()
# MAGIC             # Log the original dataset that supports mlflow logging
# MAGIC             fe.log_model(
# MAGIC                 model=clf,
# MAGIC                 artifact_path="decision_tree",
# MAGIC                 flavor=mlflow.sklearn,
# MAGIC                 training_set=training_set_spec_,
# MAGIC                 output_schema=output_schema,
# MAGIC                 registered_model_name=model_name_
# MAGIC             )
# MAGIC
# MAGIC         else:
# MAGIC             # Log the original dataset that supports mlflow logging
# MAGIC             mlflow.log_input(feature_df, "training_features")
# MAGIC             mlflow.log_input(response_df, "training_responses")
# MAGIC             input_example = X.iloc[[0]]
# MAGIC             mlflow.sklearn.log_model(
# MAGIC                 sk_model=clf,
# MAGIC                 artifact_path="ml_model",
# MAGIC                 input_example=input_example,
# MAGIC                 registered_model_name=model_name_,
# MAGIC             )
# MAGIC         # Assign alias if provided
# MAGIC         if model_alias:
# MAGIC             time.sleep(8)  # Shorter wait time before updating alias
# MAGIC             latest_version = get_latest_model_version(model_name_)
# MAGIC             client.set_registered_model_alias(model_name_, model_alias, latest_version)
# MAGIC     return clf
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Create Training Set and Fit Model for Churn Prediction
training_set_spec = fe.create_training_set(
    df=Y_train_df, #response_df
    label="Churn", #response
    feature_lookups=features,
    exclude_columns=[primary_key]
)

# Load training dataframe based on defined feature-lookup specification
training_df = training_set_spec.load_df()

# Convert data to pandas dataframes
X_train_pdf2 = training_df.drop("Churn").toPandas()
Y_train_pdf2 = training_df.select("Churn").toPandas()

fit_and_register_model(
    X_train_pdf2, 
    Y_train_pdf2, 
    model_name, 
    20,
    model_alias= "Online", 
    training_set_spec_=training_set_spec, 
    is_online=True
    )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Inspect the Lineage
# MAGIC At this point, you can navigate to the registered model within Unity Catalog and inspect the alias and lineage.
# MAGIC > The alias was configured using the MLflow client while the lineage was generated using the Databricks SDK (`FeatureEngineeringClient()`).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5: Deploy the Model with Online Features
# MAGIC
# MAGIC Now that we have a model registered to Unity Catalog, we can deploy the model with Mosaic AI Model Serving and use the online table at the time of inference.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Set endpoint name

# COMMAND ----------

# DBTITLE 1,Retrieve Online Model Version with Unique Endpoint Name
fs_endpoint_name_online = f"ML_AS_03_Demo4_FS_{DA.unique_name('_')}"
fs_model_version = client.get_model_version_by_alias(name=model_name, alias="Online").version

# COMMAND ----------

# MAGIC %md
# MAGIC #### Configure the endpoint

# COMMAND ----------

# DBTITLE 1,Configure Endpoint with Model Details
from databricks.sdk.service.serving import EndpointCoreConfigInput
fs_endpoint_config_dict = {
    "served_models": [
        {
            "model_name": model_name,
            "model_version": fs_model_version,
            "scale_to_zero_enabled": True,
            "workload_size": "Small"
        }
    ]
}

fs_endpoint_config = EndpointCoreConfigInput.from_dict(fs_endpoint_config_dict)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Serve the endpoint

# COMMAND ----------

# DBTITLE 1,Creating or Validating Serving Endpoint
try:
  w.serving_endpoints.create_and_wait(
    name=fs_endpoint_name_online,
    config=fs_endpoint_config,
    tags=[EndpointTag.from_dict({"key": "db_academy", "value": "serve_fs_model_example"})]
  )
  
  print(f"Creating endpoint {fs_endpoint_name_online} with models {model_name} versions {fs_model_version}")

except Exception as e:
  if "already exists" in e.args[0]:
    print(f"Endpoint with name {fs_endpoint_name_online} already exists")

  else:
    raise(e)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6: Query the Model Serving Endpoint
# MAGIC We'll now query the served model using the Databricks SDK like we showed in Part 1 with the offline features table. 
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Select First 1000 Customer Records as Dictionary
dataframe_records_lookups_only = X_train_df.select('customerID').limit(1000).toPandas().to_dict(orient='records')

# COMMAND ----------

# DBTITLE 1,Count and Display FS Inference Prediction Results
import pandas as pd
from collections import Counter

print("FS Inference results:")
query_response = w.serving_endpoints.query(
    name=fs_endpoint_name_online, 
    dataframe_records=dataframe_records_lookups_only
)

# Count occurrences of "Yes" and "No" in predictions from list query_response.predictions
prediction_counts = Counter(query_response.predictions)

# Convert counts to a Pandas DataFrame
df_counts = pd.DataFrame.from_dict(prediction_counts, orient='index', columns=['Count']).reset_index()
df_counts.rename(columns={'index': 'Prediction'}, inplace=True)

# Display the DataFrame
display(df_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC This demonstration discussed how to deploy and serve machine learning models in real-time using Databricks Model Serving. It covered the differences between offline and online feature tables, configuring a model serving endpoint, and leveraging feature lookups for real-time inference. Additionally, it explores techniques for on-demand feature computation, and A/B testing with real-time model serving.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Appendix
# MAGIC Below is some additional information regarding offline and online feature tables.
# MAGIC ### More on Offline and Online Tables with Real-Time Model Serving
# MAGIC #### Offline
# MAGIC - You can use an existing Delta table in Unity Catalog that includes a primary key constraint as a feature table. If the table does not have a primary key defined, you must update the table using ALTER TABLE DDL statements to add the constraint. See Use an existing Delta table in Unity Catalog as a feature table. 
# MAGIC - Any streaming table or materialized view in Unity Catalog with a primary key can be a feature table in Unity Catalog, and you can use the Features UI and API with the table.
# MAGIC - You can update a feature table in Unity Catalog by adding new features or by modifying specific rows based on the primary key.
# MAGIC
# MAGIC Additional Reading: [Working with feature tables in UC](https://docs.databricks.com/aws/en/machine-learning/feature-store/uc/feature-tables-uc) 
# MAGIC #### Online
# MAGIC - When a scoring request comes in to the model, Model Serving automatically retrieves the published feature values needed by the model. In this way, the most recent feature values are always used for predictions. 
# MAGIC - You can create a Python UDF in a notebook or in Databricks SQL.
# MAGIC - When a Python UDF depends on the result of a FeatureLookup, the value returned if the requested lookup key is not found depends on the environment. When using score_batch, the value returned is None. When using online serving, the value returned is float("nan").
# MAGIC - Models packaged with feature metadata can be registered to Unity Catalog. The feature tables used to create the model must be stored in Unity Catalog.
# MAGIC
# MAGIC Additional Reading: [Use features in online workflows](https://docs.databricks.com/aws/en/machine-learning/feature-store/online-workflows), [Compute features on demand](https://docs.databricks.com/aws/en/machine-learning/feature-store/on-demand-features)
# MAGIC
# MAGIC ### Feature Serving - [Feature Serving Endpoints](https://docs.databricks.com/aws/en/machine-learning/feature-store/feature-function-serving)
# MAGIC
# MAGIC When you use Mosaic AI Model Serving to serve a model that was built using features from Databricks, the model automatically looks up and transforms features for inference requests. With Databricks Feature Serving, you can serve structured data for retrieval augmented generation (RAG) applications, as well as features that are required for other applications, such as models served outside of Databricks or any other application that requires features based on data in Unity Catalog.
# MAGIC Databricks Feature Serving provides a single interface that serves pre-materialized and on-demand features. It also includes the following benefits:
# MAGIC
# MAGIC - Simplicity. Databricks handles the infrastructure. With a single API call, Databricks creates a production-ready serving environment.
# MAGIC - High availability and scalability. Feature Serving endpoints automatically scale up and down to adjust to the volume of serving requests.
# MAGIC - Security. Endpoints are deployed in a secure network boundary and use dedicated compute that terminates when the endpoint is deleted or scaled to zero.
# MAGIC
# MAGIC
# MAGIC ### [FeatureSpec for Feature Serving](https://docs.databricks.com/aws/en/machine-learning/feature-store/feature-serving-tutorial#step-4-create-a-feature-spec-in-unity-catalog)
# MAGIC
# MAGIC When using **Databricks Feature Serving**, you may want to define your feature set using a `FeatureSpec`. This allows you to register a reusable bundle of feature lookups and on-demand feature functions in Unity Catalog.
# MAGIC
# MAGIC Here's an example of how to define and register a `FeatureSpec`:
# MAGIC
# MAGIC ```python
# MAGIC from databricks.feature_engineering import FeatureEngineeringClient
# MAGIC
# MAGIC # Initialize Feature Engineering client
# MAGIC fe = FeatureEngineeringClient()
# MAGIC
# MAGIC # Define the FeatureSpec name
# MAGIC feature_spec_name = f"{DA.catalog_name}.{DA.schema_name}.monthly_usage_features"
# MAGIC
# MAGIC # Create the FeatureSpec in Unity Catalog
# MAGIC try:
# MAGIC     fe.create_feature_spec(
# MAGIC         name=feature_spec_name,
# MAGIC         features=features,  # This reuses the `features` list created earlier
# MAGIC         exclude_columns=None  # Optional: use to exclude specific columns
# MAGIC     )
# MAGIC except Exception as e:
# MAGIC     if "already exists" in str(e):
# MAGIC         print(f"FeatureSpec {feature_spec_name} already exists.")
# MAGIC     else:
# MAGIC         raise e
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="blank">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy" target="blank">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use" target="blank">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/" target="blank">Support</a>

# COMMAND ----------

