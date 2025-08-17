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
# MAGIC # LAB - Model Development Tracking with **MLflow**
# MAGIC
# MAGIC In this lab, you will learn how to leverage MLflow to track and manage your model development process. First, you'll load data from a feature table and create train/test splits. Next, you'll train a classification model and track its training process using MLflow. Although MLflow supports autologging of metadata and artifacts, you will manually log parameters and artifacts to gain hands-on experience with its logging API.
# MAGIC
# MAGIC 📌 **Note:** In previous demos, we covered both supervised and unsupervised model training. In this lab, however, **you will focus exclusively on fitting and tracking a supervised model.**
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC - **Task 1:** Load the dataset from the feature store table.
# MAGIC
# MAGIC - **Task 2:** Define the model hyperparameters.
# MAGIC
# MAGIC - **Task 3:** Track the model using MLflow.
# MAGIC
# MAGIC - **Task 4:** Log a custom figure.
# MAGIC
# MAGIC - **Task 5:** Review the model details via the MLflow Experiment runs UI.
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
# MAGIC Before starting the lab, run the provided classroom setup scripts. 
# MAGIC
# MAGIC **📌 Note:** In this lab you will register MLflow models with Unity Catalog. Therefore, you will need to run the next code block to **set model registry URI to UC**. 
# MAGIC

# COMMAND ----------

# %pip install --upgrade 'mlflow-skinny[databricks]'
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Next, this script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-1.3

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
# MAGIC ## Task 1 - Load Dataset from Feature Store Table
# MAGIC
# MAGIC Use the feature store to load a dataset from a specific table.
# MAGIC    - **Load Dataset:** Utilize MLflow's `load_delta` function to seamlessly retrieve and load the dataset from the Feature Store table named **`"telco"`** in the specified catalog and schema `("DA.catalog_name" and "DA.schema_name")`. 
# MAGIC    - Convert dataset to `pandas` dataframe and explore the loaded dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Data

# COMMAND ----------

## Import the necessary library for MLflow
import mlflow

## Load the feature dataset using mlflow.data
feature_dataset = mlflow.data.load_delta(
    table_name=f"{DA.catalog_name}.{DA.schema_name}.telco",
    name="telco"
)
 
## convert the dataset to pandas df and drop the customerID column
feature_data_pd = feature_dataset.df.toPandas()
feature_data_pd = feature_data_pd.drop("customerID", axis=1)

## Convert all feature_data_pd columns to float
feature_data_pd = feature_data_pd.astype(float)

## inspect final dataset
display(feature_data_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train / Test Split
# MAGIC
# MAGIC Split the dataset into training and testing sets.

# COMMAND ----------

## Import necessary libraries
import mlflow.sklearn  # For MLflow integration
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets

## Split the dataset into training and testing sets
target_col = "Churn"
X_all = feature_data_pd.drop(labels=target_col, axis=1)
y_all = feature_data_pd[target_col]
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size=0.80, random_state=42)
print(f"We have {X_train.shape[0]} records in our training dataset")
print(f"We have {X_test.shape[0]} records in our test dataset")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2 - Define Model Hyperparameters
# MAGIC
# MAGIC In this lab, you will train a classification model. In this task define parameters for a Decision Tree Model.

# COMMAND ----------

## Define Decision Tree Classifier parameters
dtc_params = {
  'criterion': 'gini',
  'max_depth': 50,
  'min_samples_split': 20,
  'min_samples_leaf': 5
}

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 3 - Track the Model Development  with MLflow
# MAGIC
# MAGIC Initialize an MLflow run.
# MAGIC    - **Initialize MLflow Run:** Start an MLflow run to track the model development process. This allows for systematic recording of parameters, metrics, and artifacts associated with the model.
# MAGIC
# MAGIC    - **Logging Model Details:** Utilize MLflow tracking to log essential information about the model, including parameters, metrics, and other relevant artifacts.

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from mlflow.models.signature import infer_signature

## set the path for mlflow experiment
mlflow.set_experiment(f"/Users/{DA.username}/LAB-1-Model-Development-Tracking-with-MLflow")

## Turn off autologging as we want to log the model manually
mlflow.autolog(disable=True)

## Start an MLFlow run
with mlflow.start_run(run_name="Model Developing Tracking with MLflow Lab") as run:
   # Log the dataset
   mlflow.log_input(feature_dataset, context="source")
   mlflow.log_input(mlflow.data.from_pandas(X_train, source=feature_dataset.source), context="training")
   mlflow.log_input(mlflow.data.from_pandas(X_test, source=feature_dataset.source), context="test")

   ## Log parameters
   mlflow.log_params(dtc_params)

   ## Fit the model
   dtc = DecisionTreeClassifier(**dtc_params)
   dtc_mdl = dtc.fit(X_train, y_train)

   ## Define model signature
   signature = infer_signature(X_all, y_all)
    
   ## Log the model
   ## Define the model name based on the feature store catalog and schema
   model_name = f"{DA.catalog_name}.{DA.schema_name}.churnmodel"
   mlflow.sklearn.log_model(
       sk_model=dtc_mdl,
       artifact_path="model-artifacts",
       signature=signature,
       registered_model_name=model_name
   )

   ## Evaluate on the training set
   y_pred_train = dtc_mdl.predict(X_train)
   mlflow.log_metric("train_accuracy", accuracy_score(y_train, y_pred_train))
   mlflow.log_metric("train_precision", precision_score(y_train, y_pred_train))
   mlflow.log_metric("train_recall", recall_score(y_train, y_pred_train))
   mlflow.log_metric("train_f1", f1_score(y_train, y_pred_train))

   ## Evaluate on the test set
   y_pred_test = dtc_mdl.predict(X_test)
   mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred_test))
   mlflow.log_metric("test_precision",  precision_score(y_test, y_pred_test))
   mlflow.log_metric("test_recall", recall_score(y_test, y_pred_test))
   mlflow.log_metric("test_f1", f1_score(y_test, y_pred_test))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4 - Log Custom Figure
# MAGIC
# MAGIC **Log Custom Figure/Visualization:** Include the logging of a custom figure, such as a confusion matrix or any relevant visualization, to further illustrate the model's behavior. This visual representation can be valuable for model evaluation and interpretation.

# COMMAND ----------

# TODO   
## Import necessary libraries for creating and displaying a confusion matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from mlflow.client import MlflowClient
client = MlflowClient()

## Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_test, labels=[1, 0])

## Create a figure object and axes for the confusion matrix plot
fig, ax = plt.subplots(figsize=(8, 6))

## Create a ConfusionMatrixDisplay object with the computed confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 0])

## Plot the confusion matrix using the created axes and specified color map
disp.plot(cmap=plt.cm.Blues, ax=ax)

## Set the title of the plot
ax.set_title('Confusion Matrix')

## Log the confusion matrix figure to MLflow
client.log_figure(run.info.run_id, figure=fig, artifact_file="confusion_matrix.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 5 - Review model details via the UI
# MAGIC To review the model details via the MLflow UI in the Experiment runs, follow these steps:
# MAGIC
# MAGIC + Step 1: Go to the "Experiments" Section
# MAGIC
# MAGIC + Step 2: Locate Your Experiment
# MAGIC
# MAGIC + Step 3: Review Run Details
# MAGIC
# MAGIC + Step 4: Reviewing Artifacts and Model metrics
# MAGIC
# MAGIC + Step 5: Viewing Confusion Matrix Image
# MAGIC
# MAGIC + Step 6: Retrieve Model Details

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC
# MAGIC In conclusion, this lab showcased the effectiveness of MLflow in seamlessly managing the model development process. Leveraging MLflow's features, such as experiment tracking, custom metric logging, and artifact storage, enhances collaboration and ensures reproducibility. The ability to review model details through the MLflow UI provides valuable insights into model performance and aids in making informed decisions.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="blank">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy" target="blank">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use" target="blank">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/" target="blank">Support</a>

# COMMAND ----------

