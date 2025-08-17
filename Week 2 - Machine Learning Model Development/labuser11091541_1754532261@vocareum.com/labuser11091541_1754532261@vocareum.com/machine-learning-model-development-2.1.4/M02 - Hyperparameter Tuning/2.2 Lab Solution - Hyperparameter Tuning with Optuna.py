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
# MAGIC # LAB - Hyperparameter Tuning with Optuna
# MAGIC
# MAGIC Welcome to the Hyperparameter Tuning with Optuna lab! In this hands-on session, you'll gain practical insights into **optimizing machine learning models using Optuna**. Throughout the lab, we'll cover key steps, from loading the dataset and creating training/test sets to **defining a hyperparameter search space and running optimization trials with Spark**. The primary objective is to equip you with the skills to fine-tune models effectively using Spark, Optuna, and MLflow.
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC 1. Load the dataset and create training/test sets for a scikit-learn model. 
# MAGIC 1. Define the hyperparameter search space for optimization.
# MAGIC 1. Define the optimization function to fine-tune the model.
# MAGIC 1. Run hyperparameter tuning trials. 
# MAGIC 1. Search for runs using the MLflow API and visualize all runs within the MLflow experiment.
# MAGIC 1. Identify the best run based on the model's precision value programmatically and visually.
# MAGIC 1. Register the model with Unity Catalog.

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

# MAGIC %pip install -U -qq optuna
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-2.2

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
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Dataset
# MAGIC
# MAGIC In this lab, you will be using a fictional dataset from a Telecom Company, which includes customer information. This dataset encompasses **customer demographics**, including gender, as well as internet subscription details such as subscription plans and payment methods.
# MAGIC
# MAGIC In this lab, we will create and tune a model that will predict customer churn based on the **`Churn`** field. 
# MAGIC
# MAGIC A table with all features is already created for you.
# MAGIC
# MAGIC **Table name: `customer_churn`**

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split
## load the table from Unity Catalog called custome_churn
table_name = f"{DA.catalog_name}.{DA.schema_name}.customer_churn"
## Read into a PySpark DataFrame and convert to Pandas DataFrame
diabetes_dataset = spark.read.table(table_name)
customer_pd = diabetes_dataset.drop('CustomerID').toPandas()

## split dataset between features and targets. The target variable is Churn
target_col = "Churn"
X_all = customer_pd.drop(labels=target_col, axis=1)
y_all = customer_pd[target_col]

## test / train split using 95% train/5% test
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size=0.95, random_state=42)
print(f"We have {X_train.shape[0]} records in our training dataset")
print(f"We have {X_test.shape[0]} records in our test dataset")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Define the Search Space and Optimization Function
# MAGIC
# MAGIC Define the parameter search space for Optuna.
# MAGIC
# MAGIC Your objective function should meet the following requirements:
# MAGIC
# MAGIC 1. Define the search space using the hyperparameters `max_depth` and `max_features`. For `max_depth`, the search range should be between 5 and 50, while `max_features` should be between 5 and 10. Additionally, for the `criterion` parameter, search based on `gini`, `entropy`, and `log_loss`. 
# MAGIC 1. Enable MLflow run as a nested experiment.
# MAGIC 1. For each run, log the cross-validation results for `accuracy`, `precision`, `recall`, and `f1`.
# MAGIC 1. Use **3-fold** cross-validation. Be sure to average the fold results using `.mean()`.
# MAGIC 1. The objective will be to _maximize_ **`precision`**.

# COMMAND ----------

import optuna
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from mlflow.models.signature import infer_signature

## Define the objective function
def optuna_objective_function(trial):
    params = {
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'max_features': trial.suggest_int('max_features', 5, 10)
    }
    
    with mlflow.start_run(nested=True, run_name=f"Optuna Trial {trial.number}"):
        
        ## Train model
        dtc = DecisionTreeClassifier(**params)
        dtc.fit(X_train, y_train)

        ## Perform cross-validation
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
        cv_results = cross_validate(dtc, X_train, y_train, cv=3, scoring=scoring_metrics)

        ## Create input signature using the first row of X_train
        input_example = X_train.iloc[[0]]
        signature = infer_signature(input_example, dtc.predict(input_example))

        ## Compute and log average scores
        cv_results_avg = {metric: cv_results[f'test_{metric}'].mean() for metric in scoring_metrics}
        mlflow.log_metrics(cv_results_avg)
        mlflow.log_params(params)
        mlflow.sklearn.log_model(dtc, "lab_optuna_decision_tree_model", signature = signature, input_example=input_example)

        ## Return precision to maximize it
        return cv_results_avg['precision']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create an Optuna Study and Log with MLflow
# MAGIC
# MAGIC First, we will delete all previous runs to keep our workspace and experiment tidy. Second, you will create an Optuna study and run the experiment with MLflow.

# COMMAND ----------

## Set the MLflow experiment name and get the id
experiment_name = f"/Users/{DA.username}/Lab_Optuna_Experiment_{DA.schema_name}"
print(f"Experiment Name: {experiment_name}")
mlflow.set_experiment(experiment_name)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
print(f"Experiment ID: {experiment_id}")

print("Clearing out old runs (If you want to add more runs, change the n_trial parameter in the next cell) ...")
## Get all runs
runs = mlflow.search_runs(experiment_ids=[experiment_id], output_format="pandas")

if runs.empty:
    print("No runs found in the experiment.")
else:
    ## Iterate and delete each run
    for run_id in runs["run_id"]:
        mlflow.delete_run(run_id)
        print(f"Deleted run: {run_id}")

    print("All runs have been deleted.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create the Study and Log with MLflow
# MAGIC
# MAGIC #### Instructions:
# MAGIC
# MAGIC 1. Create an Optuna study with name `lab_optuna_hpo`.
# MAGIC 1. Maximize the objective function. 
# MAGIC 1. Give the parent run the name `Lab_Optuna_Hyperparameter_Optimization`.
# MAGIC 1. Only run 10 trials with Optuna.

# COMMAND ----------

study = optuna.create_study(
    study_name="lab_optuna_hpo",
    direction="maximize"
)

with mlflow.start_run(run_name='Lab_Optuna_Hyperparameter_Optimization') as parent_run:
    ## Run optimization
    study.optimize(
        optuna_objective_function, 
        n_trials=10,
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3. Visual Inspection of Precision Values
# MAGIC
# MAGIC Here, we can view all 10 runs. After completing the code and running the following cell, scroll to the right and locate the column `metrics.precision`. Use the UI to order and order by descending. This will locate the largest precision score. Next, you will create a visual to also help understand the distribution of scores by trial. 
# MAGIC
# MAGIC
# MAGIC ### Creating a precision score visual
# MAGIC
# MAGIC 1. **Run the next cell** to generate the table output.  
# MAGIC 1. Click on the **plus (+) symbol** in the output cell.  
# MAGIC 1. Select **Visualization** from the options.  
# MAGIC 1. In the visualization settings, choose 
# MAGIC **Bar** and ensure **Horizontal Chart** toggle is **on**.  
# MAGIC 1. Configure the **Y-axis**:  
# MAGIC    - Set **Y Column** to `tags.mlflow.runName`.  
# MAGIC 1. Configure the **X-axis**:  
# MAGIC    - Set **X Columns** to `metrics.precision`.  
# MAGIC    - Choose **Sum** as the aggregation method.  
# MAGIC 1. Click on the **Y-axis tab**:  
# MAGIC    - Ensure **Show Labels** is **on**.  
# MAGIC 1. Apply the settings and visualize the data.
# MAGIC
# MAGIC
# MAGIC After following the above instructions, visually inspect which trial had the best run according to `precision`.

# COMMAND ----------

import mlflow
import pandas as pd

## Define your experiment name or ID
experiment_id = parent_run.info.experiment_id # Replace with your actual experiment ID

## Fetch all runs from the experiment
df_runs = mlflow.search_runs(
  experiment_ids=[experiment_id]
  )

display(df_runs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4. Find the Best Run Programmatically
# MAGIC
# MAGIC In this step you will find the best scores using the Optuna library to find the best value and parameter values. Additionally, you will use MLflow to find these values. 
# MAGIC
# MAGIC #### Instructions
# MAGIC 1. Use the Optuna study to find the best precision score. 
# MAGIC 1. Use the Optuna study to find the best hyperparameter values. 
# MAGIC 1. Use the MLflow API to find the best run based on precision score.

# COMMAND ----------

## Display the best hyperparameters and metric
print(f"Best hyperparameters: {study.best_params}")
print(f"Best precision score: {study.best_value}")

# COMMAND ----------

search_runs_pd = (mlflow.search_runs(
    experiment_ids=[experiment_id],
    order_by=["metrics.precision DESC"],
    max_results=1))

## convert search_runs_pd to pyspark dataframe
search_runs_sd = spark.createDataFrame(search_runs_pd)
display(search_runs_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the Best Model and Parameters and Register to Unity Catalog
# MAGIC
# MAGIC #### Instructions:
# MAGIC 1. Either use the results from above to copy and paste the run_id and experiment_id below or perform this task programmatically using `.collect()` on the `search_runs` PySpark DataFrame. 
# MAGIC 1. Load the model from MLflow.
# MAGIC 1. Display the results for the best model and parameters.

# COMMAND ----------

## Get the string value from run_id and experiment_id from PySpark DataFrame hpo_runs_df
run_id = search_runs_sd.select("run_id").collect()[0][0]
experiment_id = search_runs_sd.select("experiment_id").collect()[0][0]

print(f"Run ID: {run_id}")
print(f"Experiment ID: {experiment_id}")

# COMMAND ----------

import mlflow
import json
from mlflow.models import Model

## Grab an input example from the test set
input_example = X_test.iloc[[0]]

model_path = f"dbfs:/databricks/mlflow-tracking/{experiment_id}/{run_id}/artifacts/lab_optuna_decision_tree_model"

## Load the model using the run ID
loaded_model = mlflow.pyfunc.load_model(model_path)

## Retrieve model parameters MLflow client and get_run() method
client = mlflow.tracking.MlflowClient()
params = client.get_run(run_id).data.params

## Display model parameters
print("Best Model Parameters:")
print(json.dumps(params, indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the Model to Unity Catalog
# MAGIC
# MAGIC Register your model to Unity Catalog under the name `lab_optuna_model`. 
# MAGIC
# MAGIC > _You can get the catalog name and schema name using `DA.catalog_name` and `DA.schema_name`, respectively._

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
model_uri = f'runs:/{run_id}/lab_optuna_decision_tree_model'
mlflow.register_model(model_uri=model_uri, name=f"{DA.catalog_name}.{DA.schema_name}.lab_optuna_model")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab, you learned about Optuna and how to integrate Optuna trials and studies with MLflow. You also demonstrated the ability to programmatically and visually inspect the best trial. Finally, you showed how to load the MLflow model and register it to Unity Catalog.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="blank">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy" target="blank">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use" target="blank">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/" target="blank">Support</a>

# COMMAND ----------

