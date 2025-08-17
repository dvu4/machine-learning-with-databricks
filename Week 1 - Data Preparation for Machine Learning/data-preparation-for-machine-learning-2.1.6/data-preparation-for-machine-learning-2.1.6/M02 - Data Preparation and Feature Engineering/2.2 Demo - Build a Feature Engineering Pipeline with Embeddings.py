# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Demo - Build a Feature Engineering Pipeline with Embeddings
# MAGIC
# MAGIC In this demo, we will build a feature engineering pipeline that performs data loading, imputation, transformation, and embedding generation for categorical features. The pipeline will be applied to training and testing datasets, ensuring consistency in data preprocessing. Finally, we will save the pipeline for future reuse, allowing efficient and reproducible data preparation for machine learning.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to:*
# MAGIC
# MAGIC * Build a structured feature engineering pipeline that includes multiple preprocessing steps.
# MAGIC * Create a pipeline with tasks for data imputation and numerical feature scaling.
# MAGIC * Generate embeddings for categorical features to represent categorical data effectively.
# MAGIC * Assemble transformed numerical and embedded categorical features into a single feature vector.
# MAGIC * Apply the feature engineering pipeline to both training and test datasets.
# MAGIC * Display the results of the transformation.
# MAGIC * Save a data preparation and feature engineering pipeline to Unity Catalog for potential future use.

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

# MAGIC %run ../Includes/Classroom-Setup-2.2

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this demo, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains **variables such as your username, catalog name, schema name, working directory, and dataset locations**. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Data Preparation
# MAGIC
# MAGIC Before constructing the feature engineering pipeline, we need to ensure the dataset is consistent and properly formatted. This includes handling data types, addressing missing values, and preparing the dataset for further transformations. The `Telco Customer Churn` dataset will be used for this process.
# MAGIC
# MAGIC **Steps in Data Preparation:**
# MAGIC 1. Load the dataset into a Spark DataFrame.
# MAGIC 1. Split the dataset into training and testing sets.
# MAGIC 1. Convert Integer and Boolean columns to Double to ensure compatibility with Spark ML.
# MAGIC 1. Handle missing values by identifying and imputing them in:
# MAGIC     - Numeric columns
# MAGIC     - String columns
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Loading the Dataset
# MAGIC We start by loading the dataset from the specified file path using Spark.
# MAGIC
# MAGIC > This step ensures that only relevant columns are included for feature engineering and model training.

# COMMAND ----------

from pyspark.sql.functions import when, col

# Load dataset with spark
shared_volume_name = 'telco' # From Marketplace
csv_name = 'telco-customer-churn-missing' # CSV file name
dataset_path = f"{DA.paths.datasets.telco}/{shared_volume_name}/{csv_name}.csv" # Full path

telco_df = spark.read.csv(dataset_path, header="true", inferSchema="true", multiLine="true", escape='"')

# Select columns of interest
telco_df = telco_df.select("gender", "SeniorCitizen", "Partner", "tenure", "InternetService", "Contract", "PaperlessBilling", "PaymentMethod", "TotalCharges", "Churn")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preprocessing the Dataset
# MAGIC
# MAGIC Some columns require cleaning and type conversion to maintain consistency and compatibility with machine learning models.
# MAGIC
# MAGIC **Handling Null Values**  
# MAGIC In some datasets, missing values might be stored as the string `"null"`, which needs to be properly converted to `NULL` values.  
# MAGIC This step is included to show how to handle such cases in poorly formatted datasets.
# MAGIC
# MAGIC > 💡 **Note:** For this specific dataset, Spark already recognizes `nulls` correctly, _so this step can be safely skipped._

# COMMAND ----------

# Replace string "null" with actual NULL values
for column in telco_df.columns:
    telco_df = telco_df.withColumn(column, when(col(column) == "null", None).otherwise(col(column)))

# COMMAND ----------

# MAGIC %md
# MAGIC **Converting Data Types**
# MAGIC - The `TotalCharges` column should be cast to a **double** type since it represents a numerical feature.
# MAGIC
# MAGIC     > These conversions help in maintaining proper data representation and ensure compatibility with Spark ML.

# COMMAND ----------

# clean-up columns
telco_df = telco_df.withColumn("TotalCharges", col("TotalCharges").cast("double"))

display(telco_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Splitting the Dataset into Training and Testing Sets
# MAGIC Once the data has been cleaned, we split it into training and testing sets using an 80-20 split.
# MAGIC > Since telco_df is a PySpark DataFrame, we will use `randomSplit()`.

# COMMAND ----------

train_df, test_df = telco_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transforming the Dataset
# MAGIC To ensure that all numerical and categorical features are compatible with machine learning algorithms, we perform several transformations.
# MAGIC
# MAGIC **Convert Integer and Boolean Columns to Double**
# MAGIC
# MAGIC - Many machine learning algorithms require numeric input, so we convert all **integer and boolean** columns to **double**.
# MAGIC     > This ensures numerical consistency in the dataset.

# COMMAND ----------

from pyspark.sql.types import IntegerType, BooleanType, StringType, DoubleType
from pyspark.sql.functions import col, count, when


# Get a list of integer & boolean columns
integer_cols = [column.name for column in train_df.schema.fields if (column.dataType == IntegerType() or column.dataType == BooleanType())]

# Loop through integer columns to cast each one to double
for column in integer_cols:
    train_df = train_df.withColumn(column, col(column).cast("double"))
    test_df = test_df.withColumn(column, col(column).cast("double"))

# COMMAND ----------

# MAGIC %md
# MAGIC **Identifying Missing Values**
# MAGIC
# MAGIC Handling missing data is crucial to prevent errors and bias in machine learning models. We first check for missing values in numerical and categorical columns.
# MAGIC
# MAGIC - **Find Numeric Columns with Missing Values**

# COMMAND ----------

from pyspark.sql.functions import count, when

# Identify numeric columns
num_cols = [c.name for c in train_df.schema.fields if c.dataType == DoubleType()]

# Count missing values in numeric columns
num_missing_values_logic = [count(when(col(column).isNull(), column)).alias(column) for column in num_cols]
row_dict_num = train_df.select(num_missing_values_logic).first().asDict()
num_missing_cols = [column for column in row_dict_num if row_dict_num[column] > 0]

print(f"Numeric columns with missing values: {num_missing_cols}")

# COMMAND ----------

# MAGIC %md
# MAGIC - **Find String Columns with Missing Values**

# COMMAND ----------

# Identify string columns
string_cols = [c.name for c in train_df.schema.fields if c.dataType == StringType()]

# Count missing values in string columns
string_missing_values_logic = [count(when(col(column).isNull(), column)).alias(column) for column in string_cols]
row_dict_string = train_df.select(string_missing_values_logic).first().asDict()
string_missing_cols = [column for column in row_dict_string if row_dict_string[column] > 0]

print(f"String columns with missing values: {string_missing_cols}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating a Feature Engineering Pipeline
# MAGIC
# MAGIC To efficiently preprocess and transform data for machine learning, we construct a **Spark ML pipeline**. This pipeline automates key preprocessing steps, ensuring **consistency** and **reproducibility** in data preparation. The pipeline processes the **Telco Customer Churn** dataset by performing:
# MAGIC
# MAGIC **Key Feature Engineering Steps:**
# MAGIC - **Generating Embeddings for Categorical Features**  
# MAGIC   - Instead of traditional encoding techniques, we generate **dense vector representations** using SparkML's **Word2Vec**.  
# MAGIC   - These embeddings capture **semantic relationships** between categories, improving model performance.  
# MAGIC   > For example, we aim to capture the relationship between a __senior citizen with fiber optic internet__ and one with DSL. By combining categorical columns into a single sequence before embedding, we allow the model to learn cross-column patterns.
# MAGIC
# MAGIC - **Handling Missing Values**  
# MAGIC   - Missing values in **numerical columns** (e.g., `tenure`, `TotalCharges`) are imputed using the **mean strategy** to ensure completeness.  
# MAGIC   - Missing **categorical values** are handled by replacing them with `"unknown"` before embedding.
# MAGIC
# MAGIC - **Standardizing Numerical Features**  
# MAGIC   - SparkML's `VectorAssembler` **combines imputed numerical columns** into a single feature vector.  
# MAGIC   - SparkML's `StandardScaler` **standardizes numerical values**, reducing sensitivity to outliers.  
# MAGIC
# MAGIC - **Combining Features into a Final Vector**  
# MAGIC   - The **scaled numerical features** and the **single categorical embedding vector** are **combined** into one final feature vector.  
# MAGIC   - This results in a **structured and uniform format** for downstream machine learning models.
# MAGIC
# MAGIC - **Encapsulating Steps into a Pipeline**  
# MAGIC   - All preprocessing steps are included in a **Spark ML pipeline**, making the transformations **modular, reusable, and production-ready**.

# COMMAND ----------

# MAGIC %md
# MAGIC ###Generating Embeddings for Categorical Features
# MAGIC
# MAGIC To improve model performance, we create **embedding vectors** for categorical columns using **Word2Vec**. These embeddings capture complex relationships between different categories.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC > 💡 **Embedding Strategy: One Combined vs. Per-Column Embeddings**
# MAGIC
# MAGIC In this demo, we generate **a single embedding vector** by combining all categorical columns into one sequence.  
# MAGIC This approach captures **interactions across multiple categorical features** (e.g., how contract type and internet service may co-occur).
# MAGIC
# MAGIC However, there are two common strategies when using embeddings:
# MAGIC
# MAGIC | Strategy | Pros | Cons |
# MAGIC |---------|------|------|
# MAGIC | **Combined Embedding (used here)** | Captures relationships across columns; <br> Smaller feature set | Less interpretable; <br>May dilute column-specific meaning |
# MAGIC | **Per-Column Embeddings** | Clear semantic meaning per column;<br> Easier to debug | Higher dimensionality; <br>Doesn’t capture cross-column interactions |
# MAGIC
# MAGIC > There’s no one-size-fits-all solution. Combined embeddings are great for models that benefit from feature interactions, while per-column embeddings may be more suitable when interpretability is important.
# MAGIC
# MAGIC Feel free to experiment with both approaches and choose based on your dataset and modeling goals.
# MAGIC

# COMMAND ----------

from pyspark.ml.feature import Word2Vec
from pyspark.sql.functions import split, concat_ws, col, when
from pyspark.ml.linalg import DenseVector, VectorUDT
import pyspark.sql.functions as F

def generate_categorical_embeddings(train_df, test_df, categorical_cols, vector_size=5):
    """
    Generate embeddings for categorical columns using Word2Vec trained on the training DataFrame,
    and apply the same model to both train and test sets.

    Parameters:
    - train_df (DataFrame): Training Spark DataFrame
    - test_df (DataFrame): Testing Spark DataFrame
    - categorical_cols (list): List of categorical column names
    - vector_size (int): Size of the embedding vectors

    Returns:
    - Transformed train_df and test_df with embeddings as a single vector column
    """

    # Replace NULL categorical values with "unknown"
    for col_name in categorical_cols:
        train_df = train_df.withColumn(col_name, when(col(col_name).isNull(), "unknown").otherwise(col(col_name)))
        test_df = test_df.withColumn(col_name, when(col(col_name).isNull(), "unknown").otherwise(col(col_name)))

    # Combine all categorical columns into a single text column
    train_df = train_df.withColumn("categorical_sequence", concat_ws(" ", *categorical_cols))
    test_df = test_df.withColumn("categorical_sequence", concat_ws(" ", *categorical_cols))

    # Tokenize categorical data
    train_df = train_df.withColumn("categorical_tokens", split(col("categorical_sequence"), " "))
    test_df = test_df.withColumn("categorical_tokens", split(col("categorical_sequence"), " "))

    # Train Word2Vec model on training data
    word2vec = Word2Vec(vectorSize=vector_size, minCount=0, inputCol="categorical_tokens", outputCol="embedding_struct")
    model = word2vec.fit(train_df)

    # Apply model to both train and test sets
    train_df = model.transform(train_df)
    test_df = model.transform(test_df)

    return train_df, test_df

# COMMAND ----------

# MAGIC %md
# MAGIC ###Applying Embeddings to Categorical Features
# MAGIC We apply the embedding function to the categorical columns:

# COMMAND ----------

# Define categorical columns
categorical_columns = ["gender", "Partner", "InternetService", "Contract", "PaperlessBilling", "PaymentMethod"]

# Generate embeddings for categorical columns (fit on train, transform both)
train_df, test_df = generate_categorical_embeddings(train_df, test_df, categorical_columns)

# Display result
display(train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Converting Embeddings into Dense Vectors (Optional)
# MAGIC
# MAGIC For compatibility with Spark ML models, we ensure that embedding vectors are in `DenseVector` format.
# MAGIC
# MAGIC > **Note:** Spark's `Word2Vec` model already produces embeddings as `DenseVector` by default.  
# MAGIC > 
# MAGIC > In this step, we demonstrate how to manually convert embeddings into `DenseVector` format, a useful pattern when working with transformations that may output sparse vectors (e.g., one-hot encoders or feature hashing).

# COMMAND ----------

# Convert the single embedding_struct column to DenseVector
DenseVector_udf = F.udf(lambda v: DenseVector(v.values) if v else DenseVector([0.0] * 5), VectorUDT())
train_df = train_df.withColumn("categorical_embedding", DenseVector_udf(F.col("embedding_struct")))
test_df = test_df.withColumn("categorical_embedding", DenseVector_udf(F.col("embedding_struct")))

# Drop temporary columns
train_df = train_df.drop("categorical_sequence", "categorical_tokens", "embedding_struct")
test_df = test_df.drop("categorical_sequence", "categorical_tokens", "embedding_struct")

display(train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Feature Engineering and Pipeline Initialization
# MAGIC Now that categorical columns have been transformed into embeddings, we finalize the feature engineering steps.
# MAGIC
# MAGIC - **Handle Missing Values in Numerical Columns**
# MAGIC   - Impute missing numerical values with the mean of each column.
# MAGIC - **Standardize Numerical Features**
# MAGIC   - Use `StandardScaler` to normalize numerical features, reducing sensitivity to outliers.
# MAGIC - **Assemble the Final Feature Vector**
# MAGIC   - Combine numerical and categorical embeddings into a single feature vector.
# MAGIC - **Initializing the Spark ML Pipeline**
# MAGIC   - Encapsulate all transformations into a Spark ML Pipeline for structured data processing.

# COMMAND ----------

from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

# Define numerical columns for imputation
numerical_cols = ["SeniorCitizen", "tenure", "TotalCharges"]

# Impute missing numerical features
imputer = Imputer(inputCols=numerical_cols, outputCols=[col + "_imputed" for col in numerical_cols])

# Assemble numerical columns into a single vector
numerical_assembler = VectorAssembler(inputCols=[col + "_imputed" for col in numerical_cols], outputCol="numerical_assembled")

# Scale numerical features to standardize values
numerical_scaler = StandardScaler(inputCol="numerical_assembled", outputCol="numerical_scaled")

# Assemble all features (numerical + single embedding vector) into a single feature vector
feature_cols = ["numerical_scaled", "categorical_embedding"]
vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="all_features")

# Define the sequence of transformations
stages_list = [imputer, numerical_assembler, numerical_scaler, vector_assembler]

# Instantiate the pipeline
pipeline = Pipeline(stages=stages_list)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Fit the Pipeline
# MAGIC
# MAGIC In the context of machine learning and MLflow, **`fitting`** corresponds to the process of training a machine learning model on a specified dataset. 
# MAGIC
# MAGIC In the previous step we created a pipeline. Now, we will fit a model based on the pipeline. This pipeline will impute missing values, scale numerical columns, generate embeddings for categorical variables, and create a feature vector for modeling.

# COMMAND ----------

# MAGIC %md
# MAGIC **What Happens During Fitting?**
# MAGIC
# MAGIC When we call `.fit(train_df)`, the pipeline applies the following transformations:
# MAGIC
# MAGIC - **Imputation of Missing Values**
# MAGIC   - The `Imputer` calculates the **mean** for numerical columns in `train_df` and replaces missing values accordingly.
# MAGIC   
# MAGIC - **Scaling of Numerical Features**
# MAGIC   - The `StandardScaler` computes **scaling factors** based on the distribution of numerical features.
# MAGIC   - These factors are applied uniformly across datasets to **normalize feature values**.
# MAGIC
# MAGIC - **Generating Embeddings for Categorical Variables**
# MAGIC   - The **Word2Vec model** converts categorical text data into **dense vector representations**.
# MAGIC   - These embeddings **capture semantic relationships** between categories.
# MAGIC   - The trained embedding model is **stored** and later applied to unseen data.
# MAGIC
# MAGIC - **Combining Features into a Single Vector**
# MAGIC   - The `VectorAssembler` consolidates:
# MAGIC   
# MAGIC     Machine learning models in Spark ML require **all features to be represented as a single vector**.
# MAGIC     - **Scaled numerical features** - standardized values from `StandardScaler`.
# MAGIC     - **Categorical embeddings** - Dense embeddings generated by Word2Vec.
# MAGIC   - This results in a **final feature vector**, ready for input into machine learning models.

# COMMAND ----------

# Fit the Pipeline
pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Apply the Feature Engineering Pipeline
# MAGIC
# MAGIC Once the pipeline is **fitted** to the training data, it can be **applied to any dataset** using `.transform()`.  
# MAGIC We apply the pipeline to both:
# MAGIC - **Train Dataset (`train_df`)** → Generates **transformed training features**.
# MAGIC - **Test Dataset (`test_df`)** → Ensures that the same transformations are applied consistently.
# MAGIC
# MAGIC The output is a **transformed dataset** with the **final feature vector** ready for modeling.
# MAGIC

# COMMAND ----------

# Transform both training_df and test_df
train_transformed_df = pipeline_model.transform(train_df)
test_transformed_df = pipeline_model.transform(test_df)

# COMMAND ----------

# Show transformed features
train_transformed_df.select("all_features").show(3, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save and Reuse the Pipeline
# MAGIC
# MAGIC Preserving the Telco Customer Churn Prediction pipeline, encompassing the model, parameters, and metadata, is vital for maintaining reproducibility, enabling version control, and facilitating collaboration among team members. This ensures a detailed record of the machine learning workflow. In this section, we will follow these steps;
# MAGIC
# MAGIC 1. **Save the Pipeline:** Save the pipeline model, including all relevant components, to the designated artifact storage. The saved pipeline is organized within the **`spark_pipelines`** folder for clarity.
# MAGIC
# MAGIC 1. **Explore Loaded Pipeline Stages:** Upon loading the pipeline, inspect the stages to reveal key transformations and understand the sequence of operations applied during the pipeline's execution.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Save the Pipeline

# COMMAND ----------

# Save the pipeline model with overwrite mode
pipeline_model.write().overwrite().save(f"{DA.paths.working_dir}/spark_pipelines")
print(f"Saved model to: {DA.paths.working_dir}/spark_pipelines")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load and Use Saved Model

# COMMAND ----------

# Load and use the saved model
from pyspark.ml import PipelineModel

loaded_pipeline = PipelineModel.load(f"{DA.paths.working_dir}/spark_pipelines")

# Show pipeline stages
loaded_pipeline.stages

# COMMAND ----------

# MAGIC %md
# MAGIC > **Using Saved Pipeline for Reuse**  
# MAGIC > 
# MAGIC > Although we already applied the pipeline earlier in this demo, we reload the saved pipeline and apply it again here to illustrate how saved pipelines can be reused in production.

# COMMAND ----------

# Use the loaded pipeline to transform the test dataset
test_transformed_df = loaded_pipeline.transform(test_df)
display(test_transformed_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demo, we built a feature engineering pipeline to streamline data preparation. The pipeline handled data loading, missing value imputation, numerical feature scaling, and generated embeddings for categorical variables using `Word2Vec`. 
# MAGIC
# MAGIC By applying the pipeline to both training and test sets, we ensured a consistent and reproducible feature transformation process. Finally, saving the pipeline allows for future reuse, enabling efficient and standardized data preprocessing for machine learning tasks.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="blank">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy" target="blank">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use" target="blank">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/" target="blank">Support</a>