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
# MAGIC # LAB - AutoML
# MAGIC
# MAGIC Welcome to the AutoML Lab! In this lab, you will explore the capabilities of AutoML using the Databricks AutoMl UI and AutoML API. 
# MAGIC
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC In this lab, you will need to complete the following tasks;
# MAGIC
# MAGIC * **Task 1 :** Load data set.
# MAGIC
# MAGIC * **Task 2 :** Create a classification experiment using the AutoML UI.
# MAGIC
# MAGIC * **Task 3 :** Create a classification experiment with the AutoML API using a feature table.
# MAGIC
# MAGIC * **Task 4 :** Retrieve the best run and show the model URI.
# MAGIC
# MAGIC * **Task 5 :** Import the notebook for a run.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## REQUIRED - SELECT CLASSIC COMPUTE
# MAGIC Before executing cells in this notebook, please select your classic compute cluster in the lab. Be aware that **Serverless** is enabled by default.
# MAGIC Follow these steps to select the classic compute cluster:
# MAGIC 1. Navigate to the top-right of this notebook and click the drop-down menu to select your cluster. By default, the notebook will use **Serverless**.
# MAGIC 1. If your cluster is available, select it and continue to the next cell. If the cluster is not shown:
# MAGIC    - In the drop-down, select **More**.
# MAGIC    - In the **Attach to an existing compute resource** pop-up, select the first drop-down. You will see a unique cluster name in that drop-down. Please select that cluster.
# MAGIC   
# MAGIC **NOTE:** If your cluster has terminated, you might need to restart it in order to select it. To do this:
# MAGIC 1. Right-click on **Compute** in the left navigation pane and select *Open in new tab*.
# MAGIC 1. Find the triangle icon to the right of your compute cluster name and click it.
# MAGIC 1. Wait a few minutes for the cluster to start.
# MAGIC 1. Once the cluster is running, complete the steps above to select your cluster.

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
# MAGIC Before starting the lab, run the provided classroom setup script. This script will define configuration variables necessary for the lab. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-3.2

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
# MAGIC ## Task 1 : Load data set
# MAGIC
# MAGIC Load the dataset that will be used for the AutoML experiment.
# MAGIC
# MAGIC * Load and display the dataset where the table name is **`bank_loan`**.
# MAGIC
# MAGIC * Load and display the feature table where the feature table is **`bank_loan_features`**.

# COMMAND ----------

loan_data = spark.read.table(f"{DA.catalog_name}.{DA.schema_name}.bank_loan")
display(loan_data)

# COMMAND ----------

loan_features_data = spark.read.table(f"{DA.catalog_name}.{DA.schema_name}.bank_loan_features")
display(loan_features_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Create Classification Experiment Using AutoML UI
# MAGIC
# MAGIC Follow these steps to create an AutoML experiment using the  UI:
# MAGIC
# MAGIC   ***Step 1.*** Navigate to the **Experiments** section.
# MAGIC
# MAGIC   ***Step 2.*** Click on **Classification**.
# MAGIC
# MAGIC   ***Step 3.*** Choose a cluster for experiment execution.
# MAGIC
# MAGIC   ***Step 4.*** Select the input training dataset as **`catalog > database > bank_loan`**.
# MAGIC
# MAGIC   ***Step 5.*** Specify **`Personal_Loan`** as the prediction target.
# MAGIC
# MAGIC   ***Step 6.*** Deselect the **`ID`**, **`ZIP_Code`** field as it's not needed as a feature.
# MAGIC
# MAGIC   ***Step 7.*** Enter a name for your experiment, like `Bank_Loan_Prediction_AutoML_Experiment`.
# MAGIC
# MAGIC   ***Step 8.*** In the **Advanced Configuration** section, set the **Timeout** to **5 minutes**.
# MAGIC
# MAGIC   ***Step 9.*** Click on **Start AutoML**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Create a Classification Experiment with the AutoML API
# MAGIC
# MAGIC Utilize the AutoML API to set up and run a classification experiment. Follow these steps:
# MAGIC
# MAGIC 1. **Setting up the Experiment:**
# MAGIC    
# MAGIC    - **Specify the Dataset:** Specify the dataset using the Spark table name, which is **`bank_loan`**.
# MAGIC    
# MAGIC    - **Set Target Column:** Assign the target_col to the column you want to predict, which is **`Personal_Loan`**.
# MAGIC    
# MAGIC    - **Adjust Exclude Columns:** Provide a list of columns to exclude from the modeling process after reviewing the displayed dataset.
# MAGIC    
# MAGIC    - **Use features table** to be used as part of training. Feature table name: **`bank_loan_features`**.
# MAGIC    
# MAGIC    - **Set Timeout Duration:** Determine the timeout_minutes for the AutoML experiment. such as `5` minutes.   
# MAGIC
# MAGIC 2. **Running AutoML:**
# MAGIC    - Use the AutoML API to explore various machine learning models.
# MAGIC
# MAGIC

# COMMAND ----------

from databricks import automl
from datetime import datetime

features_table_path = f"{DA.catalog_name}.{DA.schema_name}.bank_loan_features"

## Define the feature store lookups
feauture_store_lookups = [
    {
        "table_name": features_table_path,
        "lookup_key": ["ID"]
    }
] 

summary = automl.classify(
    dataset = loan_data,
    target_col = "Personal_Loan",
    exclude_cols =["ID", "ZIP_Code"], 
    timeout_minutes = 5,
    feature_store_lookups = feauture_store_lookups
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Retrieve the best run and show the model URI
# MAGIC
# MAGIC Identify the best model generated by AutoML based on a chosen metric. Retrieve information about the best run, including the model URI, to further explore and analyze the model.
# MAGIC  + Find the experiment id associated with your AutoML run experiment. 
# MAGIC  + Define a search term to filter for runs. Adjust the search term based on the desired status, such as `FINISHED` or `ACTIVE`. 
# MAGIC  + Specify the run view type to view only active runs or to view all runs.
# MAGIC  + Provide the metric you want to use for ordering  and Specify whether you want to order the runs in descending or ascending order.

# COMMAND ----------

import mlflow
from mlflow.entities import ViewType

## Find the best run ...
automl_runs_pd = mlflow.search_runs(
  experiment_ids=[summary.experiment.experiment_id], 
  filter_string=f"attributes.status = 'FINISHED'", 
  run_view_type=ViewType.ACTIVE_ONLY, 
  order_by=["metrics.val_f1_score DESC"] 
)

# COMMAND ----------

automl_runs_pd.display()

# COMMAND ----------

## Print information about the best trial
print(summary.best_trial)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Task 5: Import Notebook for a Run
# MAGIC
# MAGIC AutoML automatically generates the best run's notebook and makes it available for you. If you want to access to other runs' notebooks, you need to import them.
# MAGIC
# MAGIC In this task, you will import the **5th run's notebook** to the **`destination_path`**. 
# MAGIC
# MAGIC Show the `url` and `path` of the imported notebook.

# COMMAND ----------

destination_path = f"/Users/{DA.username}/imported_notebooks/lab.3-{datetime.now().strftime('%Y%m%d%H%M%S')}"

## Get the path and url for the generated notebook
result = automl.import_notebook(summary.trials[1].artifact_uri, destination_path)
print(result.path)
print(result.url)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab, you got hands-on with Databricks AutoML. You started by loading a dataset and creating a classification experiment using the AutoMl UI and AutoML API. You then learned how to summarize the best model by applying specific filters and explored the process of retrieving the best model along with its Model URI.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="blank">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy" target="blank">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use" target="blank">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/" target="blank">Support</a>