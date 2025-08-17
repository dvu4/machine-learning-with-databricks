# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
import mlflow
from mlflow.types.utils import _infer_schema
from mlflow.models import infer_signature
from sklearn.tree import DecisionTreeClassifier
import time
import warnings
from databricks.sdk.service.serving import EndpointTag

# COMMAND ----------

@DBAcademyHelper.add_init
def initialize_uc(self):

    table_name = 'telco_table'
    features_table_name = 'features'
    payload_table_name = 'db_academy_payload'
    spark.sql(f"USE CATALOG {DA.catalog_name}")
    spark.sql(f"USE SCHEMA {DA.schema_name}")
    spark.sql(f"DROP TABLE IF EXISTS {table_name}")
    spark.sql(f"DROP TABLE IF EXISTS {features_table_name}")
    spark.sql(f"DROP TABLE IF EXISTS {features_table_name}")
    spark.sql(f"DROP TABLE IF EXISTS {payload_table_name}")

    print(f'Using catalog {DA.catalog_name} and schema {DA.schema_name}.')

# COMMAND ----------


def set_experiment_path():
    """
    This sets the experiment path to be used for logging exeperiments for demonstration 4.1
    """
    experiment_path = f"/Users/{DA.username}/4_1_Real-Time-Deployment"
    return experiment_path

# COMMAND ----------


def delete_runs_in_experiment(experiment_id):
    """
    Deletes all runs in the specified experiment.
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    import time
    client = MlflowClient()
    runs = client.search_runs(experiment_ids=[experiment_id], max_results=10000)
    # Experiment location
    
    while runs:
        print(f"Deleting {len(runs)} runs from experiment ID {experiment_id}...")
        for run in runs:
            try:
                client.delete_run(run.info.run_id)
            except Exception as e:
                print(f"Error deleting run {run.info.run_id}: {e}")
        
        # Wait to avoid overwhelming the server
        time.sleep(1)
        
        # Fetch remaining runs
        runs = client.search_runs(experiment_ids=[experiment_id], max_results=10000)


@DBAcademyHelper.add_init
def delete_experiment_by_name(self):
    """
    Deletes a specific experiment and its associated runs by name.
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    experiment_name = set_experiment_path()
    experiment = client.get_experiment_by_name(experiment_name)
    
    print(f"Cleaning up past runs from experiment '{experiment_name}'...")
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found. Experiment {experiment_name} will be a new MLflow experiment in {experiment_name}")
        return
    
    experiment_id = experiment.experiment_id
    print(f"Found experiment '{experiment_name}' (ID: {experiment_id})")

    # Delete all runs in the experiment
    delete_runs_in_experiment(experiment_id)

    # Delete the experiment
    try:
        mlflow.delete_experiment(experiment_id)
        print(f"Deleted experiment '{experiment_name}' (ID: {experiment_id})")
    except Exception as e:
        print(f"Error deleting experiment {experiment_id}: {e}")

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

# Initialize the MLflow Client
client = MlflowClient()
@DBAcademyHelper.add_init
def clean_up_model_versions(self):

    # Define the model name
    model_name = f"{DA.catalog_name}.{DA.schema_name}.ml_model"

    def get_latest_model_version(model_name):
        """Retrieve the latest model version."""
        try:
            model_versions = client.search_model_versions(f"name = '{model_name}'")
            version_numbers = [int(mv.version) for mv in model_versions]
            return str(max(version_numbers)) if version_numbers else "1"
        except Exception as e:
            print(f"Error retrieving latest model version: {e}")
            return "1"

    # Get the latest model version
    latest_version = int(get_latest_model_version(model_name))

    # Delete all versions of the model
    for model_version in range(1, latest_version + 1):
        try:
            client.delete_model_version(name=model_name, version=str(model_version))
            print(f"Deleted model version {model_version} of {model_name}")
        except Exception as e:
            print(f"Skipping version {model_version}: {e}")

    # Drop the registered model from the catalog
    try:
        client.delete_registered_model(name=model_name)
        print(f"Deleted registered model {model_name} from the catalog.")
    except Exception as e:
        print(f"Failed to delete registered model {model_name}: {e}")

# COMMAND ----------

@DBAcademyHelper.add_init
def data_preparation(self):
    from pyspark.sql.functions import col
    from databricks.feature_engineering import FeatureLookup, FeatureEngineeringClient

    # Load dataset with spark
    shared_volume_name = 'telco' # From Marketplace
    csv_name = 'telco-customer-churn-missing' # CSV file name
    dataset_p_telco = f"{DA.paths.datasets.telco}/{shared_volume_name}/{csv_name}.csv" # Full path

    # Dataset specs
    primary_key = "customerID"
    response = "Churn"
    features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"] # Keeping numerical only for simplicity and demo purposes


    # Read dataset (and drop nan)
    # Convert all fields to double for spark compatibility
    telco_df = spark.read.csv(dataset_p_telco, inferSchema=True, header=True, multiLine=True, escape='"')\
                .withColumn("TotalCharges", col("TotalCharges").cast('double'))\
                .withColumn("SeniorCitizen", col("SeniorCitizen").cast('double'))\
                .withColumn("Tenure", col("tenure").cast('double'))\
                .na.drop(how='any')

    # Separate features and ground-truth
    features_df = telco_df.select(primary_key, *features)
    response_df = telco_df.select(primary_key, response)

    # Write features_df to an offline feature table
    feature_table_name = f"{DA.catalog_name}.{DA.schema_name}.features"
    response_table_name = f"{DA.catalog_name}.{DA.schema_name}.response"
    features_for_online_table_name = f"{DA.catalog_name}.{DA.schema_name}.features_for_online"
    

    spark.sql(f'DROP TABLE IF EXISTS {feature_table_name}') # Drop existing feature table if it exists
    spark.sql(f'DROP TABLE IF EXISTS {response_table_name}') # Drop existing response table if it exists

    # Create feature table
    fe = FeatureEngineeringClient()
    fe.create_table(
        name=feature_table_name,
        df=features_df,
        primary_keys=[primary_key],
        description="Example feature table"
    )

    # Create response table
    response_df.write.mode("overwrite").saveAsTable(response_table_name)

    return f"Created features table in {feature_table_name}"

# COMMAND ----------

# from databricks.feature_engineering import FeatureEngineeringClient
# import mlflow
# from mlflow.types.utils import _infer_schema
# from mlflow.models import infer_signature
# from sklearn.tree import DecisionTreeClassifier
# import time
# import warnings
# from databricks.sdk.service.serving import EndpointTag


# def get_latest_model_version(model_name):
#     """Retrieve the latest model version."""
#     model_versions = client.search_model_versions(f"name = '{model_name}'")
#     version_numbers = [int(mv.version) for mv in model_versions]
#     return str(max(version_numbers)) if version_numbers else "1"

# def fit_and_register_model(X, Y, model_name_, random_state_, model_alias=None, log_with_fs=False, training_set_spec_=None):
#     """Train and register a Decision Tree model."""
#     clf = DecisionTreeClassifier(random_state=random_state_)

#     with mlflow.start_run(run_name=f"Train_DecisionTree_{random_state_}"):
#         mlflow.sklearn.autolog(
#             log_input_examples=True,
#             log_models=False,
#             log_post_training_metrics=True,
#             silent=True
#         )

#         clf.fit(X, Y)

#         # Log model
#         if log_with_fs:
#             try:
#                 output_schema = _infer_schema(Y)
#             except Exception as e:
#                 warnings.warn(f"Could not infer model output schema: {e}")
#                 output_schema = None
            
#             fe = FeatureEngineeringClient()
#             fe.log_model(
#                 model=clf,
#                 artifact_path="decision_tree",
#                 flavor=mlflow.sklearn,
#                 training_set=training_set_spec_,
#                 output_schema=output_schema,
#                 registered_model_name=model_name_
#             )
#         else:
#             mlflow.sklearn.log_model(
#                 clf,
#                 artifact_path="decision_tree",
#                 signature=infer_signature(X, Y),
#                 registered_model_name=model_name_
#             )

#         # Assign alias if provided
#         if model_alias:
#             time.sleep(5)  # Shorter wait time before updating alias
#             latest_version = get_latest_model_version(model_name_)
#             client.set_registered_model_alias(model_name_, model_alias, latest_version)

#     return clf

# @DBAcademyHelper.add_init
# def train_and_register_model(self):
#     import mlflow
#     # Initialize the MLflow Client
#     client = mlflow.MlflowClient()

#     # Define experiment path and set it explicitly
#     experiment_path = set_experiment_path()
#     mlflow.set_experiment(experiment_path)

#     # Define model name and feature table
#     model_name = f"{DA.catalog_name}.{DA.schema_name}.ml_model"
#     feature_table_name = f"{DA.catalog_name}.{DA.schema_name}.features"
#     response_table_name = f"{DA.catalog_name}.{DA.schema_name}.response"

#     # Read feature and response tables
#     # feature_df = spark.read.table(feature_table_name)
#     # response_df = spark.read.table(response_table_name)

#     feature_df = mlflow.data.load_delta(table_name=feature_table_name)
#     response_df = mlflow.data.load_delta(table_name=response_table_name)

#     # Convert data to Pandas
#     X_train_pdf = feature_df.drop("customerID", "Churn").toPandas()
#     Y_train_pdf = response_df.select("Churn").toPandas()

   

#     # Train and register models
#     model_champion = fit_and_register_model(X_train_pdf, Y_train_pdf, model_name, random_state_=42, model_alias="Champion")
#     model_challenger = fit_and_register_model(X_train_pdf, Y_train_pdf, model_name, random_state_=10, model_alias="Challenger")

# COMMAND ----------

def get_latest_model_version(model_name):
    """Retrieve the latest model version."""
    model_versions = client.search_model_versions(f"name = '{model_name}'")
    version_numbers = [int(mv.version) for mv in model_versions]
    return str(max(version_numbers)) if version_numbers else "1"

# COMMAND ----------

def fit_and_register_model(
    feature_df, 
    response_df, 
    model_name_, 
    random_state_, 
    model_alias=None, 
    training_set_spec_=None,
    is_online=False
    ):
    """Train and register a Decision Tree model."""
    clf = DecisionTreeClassifier(random_state=random_state_)
    if is_online:

        X = feature_df # pyspark dataframe
        y = response_df # pyspark dataframe

        
    else: 
        feature_pdf = feature_df.df.toPandas() #instance of an mlflow.data.Dataset
        response_pdf = response_df.df.toPandas() #instance of an mlflow.data.Dataset

        dataset = feature_pdf.merge(response_pdf, on="customerID", how="inner")
        # Prepare X and y
        X = dataset.drop(columns=["customerID", "Churn"])  # Drop unnecessary columns
        y = dataset["Churn"]

    with mlflow.start_run(run_name=f"Train_DecisionTree_{random_state_}"):
        mlflow.sklearn.autolog(
            log_input_examples=True,
            log_models=False,
            log_post_training_metrics=True,
            silent=True
        )

        clf.fit(X, y) # Fit the model
        

        # Log model
        if is_online:
            try:
                output_schema = _infer_schema(y)
            except Exception as e:
                warnings.warn(f"Could not infer model output schema: {e}")
                output_schema = None
            
            fe = FeatureEngineeringClient()
            # Log the original dataset that supports mlflow logging
            fe.log_model(
                model=clf,
                artifact_path="decision_tree",
                flavor=mlflow.sklearn,
                training_set=training_set_spec_,
                output_schema=output_schema,
                registered_model_name=model_name_
            )

        else:
            # Log the original dataset that supports mlflow logging
            mlflow.log_input(feature_df, "training_features")
            mlflow.log_input(response_df, "training_responses")
            input_example = X.iloc[[0]]
            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="ml_model",
                input_example=input_example,
                registered_model_name=model_name_,
            )
        # Assign alias if provided
        if model_alias:
            time.sleep(8)  # Shorter wait time before updating alias
            latest_version = get_latest_model_version(model_name_)
            client.set_registered_model_alias(model_name_, model_alias, latest_version)
    return clf


@DBAcademyHelper.add_init
def train_and_register_model(self):
    import mlflow
    # Initialize the MLflow Client
    client = mlflow.MlflowClient()

    # Define experiment path and set it explicitly
    experiment_path = set_experiment_path()
    mlflow.set_experiment(experiment_path)

    # Define model name and table names
    model_name = f"{DA.catalog_name}.{DA.schema_name}.ml_model"
    feature_table_name = f"{DA.catalog_name}.{DA.schema_name}.features"
    response_table_name = f"{DA.catalog_name}.{DA.schema_name}.response"

    # Load feature and response datasets using mlflow.data.load_delta
    feature_df = mlflow.data.load_delta(table_name=feature_table_name)
    response_df = mlflow.data.load_delta(table_name=response_table_name)

    # Train and register models
    model_champion = fit_and_register_model(feature_df, response_df, model_name, random_state_=42, model_alias="Champion")
    model_challenger = fit_and_register_model(feature_df, response_df, model_name, random_state_=10, model_alias="Challenger")

# COMMAND ----------

# Initialize DBAcademyHelper
DA = DBAcademyHelper() 
DA.init()