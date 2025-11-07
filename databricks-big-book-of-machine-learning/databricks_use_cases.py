# Databricks notebook source
# MAGIC %md
# MAGIC ## Financial Fraud Detection using Decision Tree Machine Learning Models
# MAGIC In this notebook, we will showcase the use of decision tree ML models to perform financial fraud detection.
# MAGIC
# MAGIC Original NB : https://pages.databricks.com/rs/094-YMS-629/images/financial-fraud-detection-decision-tree.html?_ga=2.261331150.2112692442.1591844546-225663068.1585060489
# MAGIC
# MAGIC ### Source Data
# MAGIC PaySim simulates mobile money transactions based on a sample of real transactions extracted from one month of financial logs from a mobile money service implemented in an African country. The original logs were provided by a multinational company, who is the provider of the mobile financial service which is currently running in more than 14 countries all around the world.
# MAGIC
# MAGIC This synthetic dataset is scaled down 1/4 of the original dataset and it is created just for Kaggle. To load the dataset yourself, please download it to your local machine from Kaggle and then import the data via Import Data: Azure | AWS.
# MAGIC
# MAGIC
# MAGIC
# MAGIC ### Dictionary
# MAGIC This is the column definition of the referenced sythentic dataset.
# MAGIC
# MAGIC
# MAGIC | Column Name      | Description                                                                                                                        |
# MAGIC |------------------|------------------------------------------------------------------------------------------------------------------------------------|
# MAGIC | step             | maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation).                |
# MAGIC | type             | CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.                                                                                    |
# MAGIC | amount           | amount of the transaction in local currency.                                                                                       |
# MAGIC | nameOrig         | customer who started the transaction                                                                                               |
# MAGIC | oldbalanceOrg    | initial balance before the transaction                                                                                             |
# MAGIC | newbalanceOrig   | new balance after the transaction                                                                                                  |
# MAGIC | nameDest         | customer who is the recipient of the transaction                                                                                   |
# MAGIC | oldbalanceDest   | initial balance recipient before the transaction. Note that there is not information for customers that start with M (Merchants).  |
# MAGIC | newbalanceDest   | new balance recipient after the transaction. Note that there is not information for customers that start with M (Merchants).       |

# COMMAND ----------

# Including MLflow
import mlflow
import mlflow.spark

import os
print("MLflow Version: %s" % mlflow.__version__)

# COMMAND ----------

from pyspark.dbutils import DBUtils
from datetime import date, datetime, timedelta
import pyspark.sql.functions as F
from pytz import timezone
import logging

logger = logging.getLogger('FRAUD DETECTION')
logger.propagate = False # Prevent log messages from being passed to the root logger
logger.handlers = []
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# create the stderr handler
stderr_handler = logging.StreamHandler()
stderr_handler.setLevel(logging.DEBUG)
stderr_handler.setFormatter(formatter)
logger.addHandler(stderr_handler)
logger.setLevel(logging.DEBUG)
logger.info(f"Start logging ...")
logger.info(f"Spark version: {spark.version}")

# COMMAND ----------

dbutils.widgets.removeAll()

# COMMAND ----------

# Changed per environment
dbutils.widgets.text("env", "prodfix")
env = dbutils.widgets.get("env")

# databricks workspace value
dbutils.widgets.text("short_code", "gci")
short_code = dbutils.widgets.get("short_code")

# Configure storage account connection
sp_name = f"p-{env}-ds-{short_code}-svcp" 
secret_scope = f"{short_code}-secrets"
client_id_name = f"{sp_name}-client-id"
client_secret_name = f"{sp_name}-secret"

storage_access_list = ["prodfixdseus2tgcisa01"]

# Setting the spark configuration
for sa in storage_access_list:
  spark.conf.set(f"fs.azure.account.auth.type.{sa}.dfs.core.windows.net", "OAuth")
  spark.conf.set(f"fs.azure.account.oauth.provider.type.{sa}.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
  spark.conf.set(f"fs.azure.account.oauth2.client.id.{sa}.dfs.core.windows.net", dbutils.secrets.get(scope=secret_scope, key=client_id_name))
  spark.conf.set(f"fs.azure.account.oauth2.client.secret.{sa}.dfs.core.windows.net", dbutils.secrets.get(scope=secret_scope, key=client_secret_name))
  spark.conf.set(f"fs.azure.account.oauth2.client.endpoint.{sa}.dfs.core.windows.net", "https://login.microsoftonline.com/92cb778e-8ba7-4f34-a011-4ba6e7366996/oauth2/token")

# Identify the databricks workspace environment
WrkSpaceId= spark.conf.get("spark.databricks.clusterUsageTags.clusterOwnerOrgId")
 
if(WrkSpaceId=='2195848036328367'):  
  env='prod'
elif(WrkSpaceId=='1851115442125419'):  
  env='prodfix'
elif(WrkSpaceId=='6340613744706656'):
  env='qa'
 
logger.info(f"Environment: {env}")

# COMMAND ----------

# DBTITLE 1,Extract the customer source mid
fraudulent_transactions_df = spark.read.csv("/FileStore/tables/archive_01.csv", header=True, inferSchema=True, sep=",")
# fraudulent_transactions_df = spark.read.csv("/FileStore/tables/archive.csv", header=True, inferSchema=True, sep=",")
# fraudulent_transactions_df = spark.read.format("csv").option("header", True).option("inferSchema", True).option("sep", ",").load("abfss://csgci-adhoc@prodfixdseus2tgcisa01.dfs.core.windows.net/db_use_case_files/archive_01.csv.zip")

df = fraudulent_transactions_df.select("step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest")

df.display()
df.count() # 8,808, 6,362,620

# COMMAND ----------

# Review the schema of your data 
df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calculate Differences between Originating and Destination Balanaces
# MAGIC With the following PySpark DataFrame query, we will calculate the following columns:

# COMMAND ----------

# Calculate the differences between originating and destination balances
df = df.withColumn("orgDiff", df.newbalanceOrig - df.oldbalanceOrg).withColumn("destDiff", df.newbalanceDest - df.oldbalanceDest)

# Create temporary view
df.createOrReplaceTempView("financials")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Let's review the data and ask some simple questions

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## What are the type of transactions?

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Organize by Type
# MAGIC select type, count(1) from financials group by type

# COMMAND ----------

# MAGIC %md
# MAGIC ## How much money are we talking about (synthetically)?

# COMMAND ----------

# MAGIC %sql
# MAGIC select type, sum(amount) from financials group by type

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rules-based Model: Create a set of rules to identify fraud based on known cases
# MAGIC The following where clause are a set of rules to identify know fraud-based cases using SQL; i.e. rules-based model.
# MAGIC
# MAGIC - Often, financial fraud analytics start with with clauses like the where clause below
# MAGIC - Note, in reality, rules are often much larger and more complicated

# COMMAND ----------

from pyspark.sql import functions as F

# Rules to Identify Known Fraud-based
df = df.withColumn("label", 
                   F.when(
                     (
                       (df.oldbalanceOrg <= 56900) & (df.type == "TRANSFER") & (df.newbalanceDest <= 105)) | 
                       (
                         (df.oldbalanceOrg > 56900) & (df.newbalanceOrig <= 12)) | 
                           (
                             (df.oldbalanceOrg > 56900) & (df.newbalanceOrig > 12) & (df.amount > 1160000)
                           ), 1
                   ).otherwise(0))

# Calculate proportions
fraud_cases = df.filter(df.label == 1).count()
total_cases = df.count()
fraud_pct = 1.*fraud_cases/total_cases

# Provide quick statistics
print("Based on these rules, we have flagged %s (%s) fraud cases out of a total of %s cases." % (fraud_cases, fraud_pct, total_cases))

# Create temporary view to review data
df.createOrReplaceTempView("financials_labeled")

# COMMAND ----------

# MAGIC %md
# MAGIC ## How much fraud are we talking about?
# MAGIC Based on the existing rules, while 4% of the transactions are fraudulent, it takes into account of the 11% of the total amount.

# COMMAND ----------

# MAGIC %sql
# MAGIC select label, count(1) as `Transactions`, sum(amount) as `Total Amount` from financials_labeled group by label

# COMMAND ----------

# MAGIC %md
# MAGIC ## Top Origination / Destination Difference Pairs (>$1M TotalDestDiff)
# MAGIC Each bar represents a pair of entities performing a transaction

# COMMAND ----------

# MAGIC %sql
# MAGIC -- where sum(destDiff) >= 10000000.00
# MAGIC select nameOrig, nameDest, label, TotalOrgDiff, TotalDestDiff
# MAGIC   from (
# MAGIC      select nameOrig, nameDest, label, sum(OrgDiff) as TotalOrgDiff, sum(destDiff) as TotalDestDiff 
# MAGIC        from financials_labeled 
# MAGIC       group by nameOrig, nameDest, label 
# MAGIC      ) a
# MAGIC  where TotalDestDiff >= 1000000
# MAGIC  limit 100

# COMMAND ----------

# MAGIC %md
# MAGIC ## What type of transactions are associated with fraud?
# MAGIC Reviewing the rules-based model, it appears that most fraudulent transactions are in the category of Transfer and Cash_Out.

# COMMAND ----------

# MAGIC %sql
# MAGIC select type, label, count(1) as `Transactions` from financials_labeled group by type, label

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rules vs. ML model
# MAGIC Instead of creating specific rules that will change over time, can we be more precise and go to production faster by creating a ML model?
# MAGIC
# MAGIC
# MAGIC ## Decision Trees
# MAGIC Decision trees and their ensembles are popular methods for the machine learning tasks of classification and regression. Decision trees are widely used since they are easy to interpret, handle categorical features, extend to the multiclass classification setting, do not require feature scaling, and are able to capture non-linearities and feature interactions. Tree ensemble algorithms such as random forests and boosting are among the top performers for classification and regression tasks.
# MAGIC
# MAGIC Because of these facets, decision trees often perform well on top of rules-based models and are often a good starting point for fraud detection.
# MAGIC
# MAGIC
# MAGIC
# MAGIC Source: The Wise Old Tree

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create training and test datasets
# MAGIC To build and validate our generalized fraud ML model, we will initially split the data using randomSplit to create our training and test datasets.

# COMMAND ----------


# Initially split our dataset between training and test datasets
(train, test) = df.randomSplit([0.8, 0.2], seed=12345)

# Cache the training and test datasets
train.cache()
test.cache()

# Print out dataset counts
print("Total rows: %s, Training rows: %s, Test rows: %s" % (df.count(), train.count(), test.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create ML Pipeline
# MAGIC When creating an ML model, there are typically a set of repeated steps (e.g. StringIndexer, VectorAssembler, etc.). By creating a ML pipeline, we can reuse this pipeline (and all of its steps) to retrain on a new and/or updated dataset.

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier

# Encodes a string column of labels to a column of label indices
indexer = StringIndexer(inputCol = "type", outputCol = "typeIndexed")

# VectorAssembler is a transformer that combines a given list of columns into a single vector column
va = VectorAssembler(inputCols = ["typeIndexed", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "orgDiff", "destDiff"], outputCol = "features")

# Using the DecisionTree classifier model
dt = DecisionTreeClassifier(labelCol = "label", featuresCol = "features", seed = 54321, maxDepth = 5)

# Create our pipeline stages
pipeline = Pipeline(stages=[indexer, va, dt])

# COMMAND ----------

# View the Decision Tree model (prior to CrossValidator)
dt_model = pipeline.fit(train)
display(dt_model.stages[-1])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use BinaryClassificationEvaluator
# MAGIC Determine the accuracy of the model by reviewing the areaUnderPR and areaUnderROC

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Use BinaryClassificationEvaluator to evaluate our model
evaluatorPR = BinaryClassificationEvaluator(labelCol = "label", rawPredictionCol = "prediction", metricName = "areaUnderPR")
evaluatorAUC = BinaryClassificationEvaluator(labelCol = "label", rawPredictionCol = "prediction", metricName = "areaUnderROC")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup CrossValidation
# MAGIC To try out different parameters to potentially improve our model, we will use CrossValidator in conjunction with the ParamGridBuilder to automate trying out different parameters.
# MAGIC
# MAGIC Note, we are using evaluatorPR as our evaluator as the Precision-Recall curve is often better for an unbalanced distribution.

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Build the grid of different parameters
paramGrid = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [5, 10, 15]) \
    .addGrid(dt.maxBins, [10, 20, 30]) \
    .build()

# Build out the cross validation
crossval = CrossValidator(estimator = dt,
                          estimatorParamMaps = paramGrid,
                          evaluator = evaluatorPR,
                          numFolds = 3)  

pipelineCV = Pipeline(stages=[indexer, va, crossval])

# Train the model using the pipeline, parameter grid, and preceding BinaryClassificationEvaluator
cvModel_u = pipelineCV.fit(train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review Results
# MAGIC Review the areaUnderPR (area under Precision Recall curve) and areaUnderROC (area under Receiver operating characteristic) or AUC (area under curve) metrics.

# COMMAND ----------

# Build the best model (training and test datasets)
train_pred = cvModel_u.transform(train)
test_pred = cvModel_u.transform(test)

# Evaluate the model on training datasets
pr_train = evaluatorPR.evaluate(train_pred)
auc_train = evaluatorAUC.evaluate(train_pred)

# Evaluate the model on test datasets
pr_test = evaluatorPR.evaluate(test_pred)
auc_test = evaluatorAUC.evaluate(test_pred)

# Print out the PR and AUC values
print("PR train:", pr_train)
print("AUC train:", auc_train)
print("PR test:", pr_test)
print("AUC test:", auc_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confusion Matrix Code-base
# MAGIC Subsequent cells will be using the following code to plot the confusion matrix.

# COMMAND ----------

# Create confusion matrix template
from pyspark.sql.functions import lit, expr, col, column

# Confusion matrix template
cmt = spark.createDataFrame([(1, 0), (0, 0), (1, 1), (0, 1)], ["label", "prediction"])
cmt.createOrReplaceTempView("cmt")

# COMMAND ----------

# Source code for plotting confusion matrix is based on `plot_confusion_matrix` 
# via https://runawayhorse001.github.io/LearningApacheSpark/classification.html#decision-tree-classification
import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, title):
  # Clear Plot
  plt.gcf().clear()

  # Configure figure
  fig = plt.figure(1)
  
  # Configure plot
  classes = ['Fraud', 'No Fraud']
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  # Normalize and establish threshold
  normalize=False
  fmt = 'd'
  thresh = cm.max() / 2.

  # Iterate through the confusion matrix cells
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  # Final plot configurations
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label') 
  
  # Display images
  image = fig
  
  # Show plot
  #fig = plt.show()
  
  # Save plot
  fig.savefig("confusion-matrix.png")

  # Display Plot
  display(image)
  
  # Close Plot
  plt.close(fig)

# COMMAND ----------

# Create temporary view for test predictions
test_pred.createOrReplaceTempView("test_pred")

# Create test predictions confusion matrix
test_pred_cmdf = spark.sql("select a.label, a.prediction, coalesce(b.count, 0) as count from cmt a left outer join (select label, prediction, count(1) as count from test_pred group by label, prediction) b on b.label = a.label and b.prediction = a.prediction order by a.label desc, a.prediction desc")

# View confusion matrix
display(test_pred_cmdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## View Confusion Matrix
# MAGIC Let's use matplotlib and pandas to visualize our confusion matrix

# COMMAND ----------

# Convert to pandas
cm_pdf = test_pred_cmdf.toPandas()

# Create 1d numpy array of confusion matrix values
cm_1d = cm_pdf.iloc[:, 2].to_numpy()

# Create 2d numpy array of confusion matrix values
cm = np.reshape(cm_1d, (-1, 2))

# COMMAND ----------

# Plot confusion matrix  
plot_confusion_matrix(cm, "Confusion Matrix (Unbalanced Test)")

# COMMAND ----------

# Get or create experiment by name
experiment_name = "/Users/duc.vu@walgreens.com/databricks_use_cases"  # Update this path as needed
mlflow.set_experiment(experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)
mlflow_experiment_id = experiment.experiment_id

# COMMAND ----------

mlflow_experiment_id

# COMMAND ----------

# Log MLflow
with mlflow.start_run(experiment_id = mlflow_experiment_id) as run:
  # Log Parameters and metrics
  mlflow.log_param("balanced", "no")
  mlflow.log_metric("PR train", pr_train)
  mlflow.log_metric("AUC train", auc_train)
  mlflow.log_metric("PR test", pr_test)
  mlflow.log_metric("AUC test", auc_test)
  
  # Log model
  mlflow.spark.log_model(dt_model, "model")
  
  # Log Confusion matrix
  mlflow.log_artifact("confusion-matrix.png")

# COMMAND ----------

# Reset the DataFrames for no fraud (`dfn`) and fraud (`dfy`)
dfn = train.filter(train.label == 0)
dfy = train.filter(train.label == 1)

# Calculate summary metrics
N = train.count()
y = dfy.count()
p = y/N

# Create a more balanced training dataset
train_b = dfn.sample(False, p, seed = 92285).union(dfy)

# Print out metrics
print("Total count: %s, Fraud cases count: %s, Proportion of fraud cases: %s" % (N, y, p))
print("Balanced training dataset count: %s" % train_b.count())

# COMMAND ----------

# Display our more balanced training dataset
display(train_b.groupBy("label").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Update ML pipeline
# MAGIC Because we had created the ML pipeline stages in the previous cells, we can re-use them to execute it against our balanced dataset.
# MAGIC
# MAGIC Note, we are using evaluatorAUC as our evaluator as this is often better for a balanced distribution.

# COMMAND ----------

# Re-run the same ML pipeline (including parameters grid)
crossval_b = CrossValidator(estimator = dt,
                          estimatorParamMaps = paramGrid,
                          evaluator = evaluatorAUC,
                          numFolds = 3)  

pipelineCV_b = Pipeline(stages=[indexer, va, crossval_b])

# Train the model using the pipeline, parameter grid, and BinaryClassificationEvaluator using the `train_b` dataset
cvModel_b = pipelineCV_b.fit(train_b)

# COMMAND ----------

# Build the best model (balanced training and full test datasets)
train_pred_b = cvModel_b.transform(train_b)
test_pred_b = cvModel_b.transform(test)

# Evaluate the model on the balanced training datasets
pr_train_b = evaluatorPR.evaluate(train_pred_b)
auc_train_b = evaluatorAUC.evaluate(train_pred_b)

# Evaluate the model on full test datasets
pr_test_b = evaluatorPR.evaluate(test_pred_b)
auc_test_b = evaluatorAUC.evaluate(test_pred_b)

# Print out the PR and AUC values
print("PR train:", pr_train_b)
print("AUC train:", auc_train_b)
print("PR test:", pr_test_b)
print("AUC test:", auc_test_b)

# COMMAND ----------

# Create temporary view for test predictions
test_pred_b.createOrReplaceTempView("test_pred_b")

# Create test predictions confusion matrix
test_pred_b_cmdf = spark.sql("select a.label, a.prediction, coalesce(b.count, 0) as count from cmt a left outer join (select label, prediction, count(1) as count from test_pred_b group by label, prediction) b on b.label = a.label and b.prediction = a.prediction order by a.label desc, a.prediction desc")

# View confusion matrix
display(test_pred_b_cmdf)

# COMMAND ----------


# Convert to pandas
cm_b_pdf = test_pred_b_cmdf.toPandas()

# Create 1d numpy array of confusion matrix values
cm_b_1d = cm_b_pdf.iloc[:, 2].to_numpy()

# Create 2d numpy array of confusion matrix vlaues
cm_b = np.reshape(cm_b_1d, (-1, 2))

# Print out the 2d array
print(cm_b)

# COMMAND ----------

# MAGIC %md
# MAGIC ## View Decision Tree Models
# MAGIC Visually compare the differences between the unbalanced and balanced decision tree models (basd on the train and train_b datasets respectively).

# COMMAND ----------

# Extract Feature Importance
#  Attribution: Feature Selection Using Feature Importance Score - Creating a PySpark Estimator
#               https://www.timlrx.com/2018/06/19/feature-selection-using-feature-importance-score-creating-a-pyspark-estimator/
import pandas as pd

def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))

# COMMAND ----------

# View the Unbalanced Decision Tree model (prior to CrossValidator)
dt_model = pipeline.fit(train)
display(dt_model.stages[-1])

# COMMAND ----------

# Extract Feature Importance for the original unbalanced dt_model
ExtractFeatureImp(dt_model.stages[-1].featureImportances, train_pred, "features").head(10)

# COMMAND ----------

# View the Balanced Decision Tree model (prior to CrossValidator)
dt_model_b = pipeline.fit(train_b)
display(dt_model_b.stages[-1])

# COMMAND ----------

# Extract Feature Importance for the nbalanced dt_model
ExtractFeatureImp(dt_model_b.stages[-1].featureImportances, train_pred_b, "features").head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Comparing Confusion Matrices
# MAGIC Below we will compare the unbalanced and balanced decision tree ML model confusion matrices.

# COMMAND ----------

# Plot confusion matrix  
plot_confusion_matrix(cm, "Confusion Matrix (Unbalanced Test)")

# COMMAND ----------

# Plot confusion matrix  
plot_confusion_matrix(cm_b, "Confusion Matrix (Balanced Test)")

# COMMAND ----------

# Log MLflow
with mlflow.start_run(experiment_id = mlflow_experiment_id) as run:
  # Log Parameters and metrics
  mlflow.log_param("balanced", "yes")
  mlflow.log_metric("PR train", pr_train_b)
  mlflow.log_metric("AUC train", auc_train_b)
  mlflow.log_metric("PR test", pr_test_b)
  mlflow.log_metric("AUC test", auc_test_b)
    
  # Log model
  mlflow.spark.log_model(dt_model_b, "model")
  
  # Log Confusion matrix
  mlflow.log_artifact("confusion-matrix.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Observation
# MAGIC There is a significant difference for the PR and AUC values when comparing the unbalanced vs. balanced decision tree models.
# MAGIC
# MAGIC |Metric/dataset|	Unbalanced|	Balanced|
# MAGIC |------------------|------------------|------------------|
# MAGIC |PR train	|0.9537894984523128	|0.999629161563572|
# MAGIC |AUC train	|0.998647996459481	|0.9998071389056655|
# MAGIC |PR test	|0.9539170535377599	|0.9904709171789063|
# MAGIC |AUC test	|0.9984378183482442	|0.9997903902204509|
# MAGIC

# COMMAND ----------

