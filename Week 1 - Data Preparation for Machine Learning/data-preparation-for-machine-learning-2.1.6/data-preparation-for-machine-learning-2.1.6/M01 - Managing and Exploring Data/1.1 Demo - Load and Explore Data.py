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
# MAGIC # Demo - Load and Explore Data 
# MAGIC
# MAGIC The ability to efficiently handle and explore data is paramount for machine learning projects. In this demo, we'll delve into techniques such as reading from and writing to Delta tables, computing statistics for machine learning insights, and visually exploring data for a comprehensive understanding of your datasets.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC By the end of this demo, you will be able to:
# MAGIC
# MAGIC * Read data from a Delta table into a pandas DataFrame.
# MAGIC * Read a previous version of data from a Delta table.
# MAGIC * Write data from a DataFrame into a Delta table.
# MAGIC * Compute summary statistics on data for machine learning using data profiles.
# MAGIC * Compute a correlation matrix for columns/features in machine learning data.
# MAGIC * Visually explore data using built-in visualization capabilities to examine outliers, continuous variable distributions, and categorical variable distributions.
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

# MAGIC %run ../Includes/Classroom-Setup-1.1

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
print(f"Datasets Location: {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Dataset 
# MAGIC Read the `.csv` file from shared volume and store as spark dataframe `telco_df`.

# COMMAND ----------

# Load dataset with spark
shared_volume_name = 'telco' # From Marketplace
csv_name = 'telco-customer-churn-missing' # CSV file name
dataset_path = f"{DA.paths.datasets.telco}/{shared_volume_name}/{csv_name}.csv" # Full path

schema_string = """
    customerID string,
    gender string,
    SeniorCitizen double,
    Partner string,
    Dependents string,
    tenure double,
    phoneService string,
    MultipleLines string,
    internetService string,
    OnlineSecurity string,
    OnlineBackup string,
    DeviceProtection string,
    TechSupport string,
    StreamingTV string,
    StreamingMovies string,
    Contract string,
    PaperlessBilling string,
    PaymentMethod string,
    MonthlyCharges double,
    TotalCharges double,
    Churn string
"""

telco_df = spark.read.csv(dataset_path, \
                          header=True,
                          schema=schema_string,
                          multiLine=True,
                          escape='"')

display(telco_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explore Data with Summary Stats
# MAGIC
# MAGIC While using notebooks, you have various options to view summary statistics for dataset. Some of these options are:
# MAGIC
# MAGIC * using spark DataFrame's built-in method (e.g. `summary()`)
# MAGIC * using databricks' utility methods (e.g. `dbutils.data.summarize()`)
# MAGIC * using databricks' built-in data profiler/visualizations
# MAGIC * using external libraries such as `matplotlib`
# MAGIC
# MAGIC
# MAGIC In this section we will go over the Spark's and Databricks' built-in features for summarizing data. In the next section, we will explore the visualization options.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The first and simplest way is using Spark's `summary` function.

# COMMAND ----------

# Display summary statistics with spark
display(telco_df.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC Another way of displaying summary statistics is to use Databricks' `summarize` function. The **`summarize`** function automatically generates a comprehensive report for the dataframe. This report encompasses crucial statistics, data types, and the presence of missing values, providing a holistic view of the dataset
# MAGIC
# MAGIC Within this generated summary, the interactive features allow us to sort the information by various criteria:
# MAGIC
# MAGIC * Feature Order
# MAGIC * Non-Uniformity
# MAGIC * Alphabetical
# MAGIC * Amount Missing/Zero
# MAGIC
# MAGIC Furthermore, leveraging the datatype information, we can selectively filter out specific datatypes for in-depth analysis. This functionality enables us to create charts tailored to our analytical preferences, facilitating a deeper understanding of the dataframe and extracting valuable insights.

# COMMAND ----------

dbutils.data.summarize(telco_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The **`Display`** function not only facilitates the viewing of detailed dataframes but also serves as a powerful tool for visualizing our dataset according to individual preferences. Whether opting for a Pie Chart or a Bar Chart, this functionality allows for a more insightful exploration, uncovering dependencies and patterns within the dataframe features.

# COMMAND ----------

display(telco_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Data Visualization
# MAGIC
# MAGIC In this section, we will explore two different methods for exploring data with visualization tools. The first option is the Databricks' rich and interactive visualization capabilities. The second option is to use an external library when you need custom visualizations that are not available in Databricks' visualization tools.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Integrated Visualization
# MAGIC
# MAGIC Let's enhance our understanding of the Telco dataset through integrated visualizations. Below are instructions to create key visualizations that reveal patterns, relationships, and distributions in the data.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Step-by-Step Instructions for Creating a Bar Chart Using the Visualization Editor
# MAGIC
# MAGIC **Step 1: Open the Visualization Editor**  
# MAGIC   + Click the **"+"** button above your output cell and select **Visualization** from the menu to launch the visualization editor.
# MAGIC
# MAGIC **Step 2: Select Visualization Type**  
# MAGIC   + At the top of the editor, locate the **Visualization type** dropdown.<br>
# MAGIC   + Click the dropdown and select **Bar** to create a bar chart.
# MAGIC
# MAGIC **Step 3: Set X and Y Columns**  
# MAGIC   + In the **General** tab:
# MAGIC     - Find the **X column** box and select the column for your horizontal axis.
# MAGIC       - *Example*: `tenure` (to show tenure buckets).
# MAGIC     - Under **Y columns**, click to select the column for your vertical axis.
# MAGIC       - *Example*: `MonthlyCharges` (to show monthly charges).
# MAGIC
# MAGIC **Step 4: Choose Aggregation for Y Column**  
# MAGIC   + Next to your chosen Y column, use the dropdown to select an aggregation method:
# MAGIC     - **Sum**: Totals values (e.g., total monthly charges per tenure group).
# MAGIC     - **Count**: Number of records per X value.
# MAGIC     - **Average**, **Median**, **Min**, **Max**, **Standard deviation**: Other statistical summaries as required.
# MAGIC     - *Example*: Choose **Sum** to display the total monthly charges for each tenure.
# MAGIC
# MAGIC **Step 5: (Optional) Group By Column**  
# MAGIC   + In the **General** tab, use the **Group by** option to break down the bars by another column.
# MAGIC     - Choose a categorical column, such as `Contract`, `PaymentMethod`, or `gender`.
# MAGIC     - *Example*: Group by `Contract` to display side-by-side bars for each contract type within each tenure bucket.
# MAGIC
# MAGIC **Step 6: Customize Chart Settings**  
# MAGIC   + Explore the tabs at the top of the editor for further customization:
# MAGIC     - **X axis**: Change scale (linear, category), edit labels, show/hide axis.
# MAGIC     - **Y axis**: Set axis range, change scaling, edit name, show/hide axis, set start or end values.
# MAGIC     - **Series**: Customize the order, rename groups/series, choose the axis (left/right) for each series.
# MAGIC     - **Colors**: Assign colors to each group or category.
# MAGIC     - **Data labels**: Enable/disable labels, select number, percent, or datetime formatting, customize display for more clarity.
# MAGIC     - **General**: Normalize values to percentage, set handling for missing/NULL values, add error columns if needed.
# MAGIC
# MAGIC **Step 7: Preview and Adjust**  
# MAGIC   + The chart will preview automatically on the right pane.
# MAGIC   + Refine your visualization by:
# MAGIC     - Changing X/Y selections, aggregation, or grouping.
# MAGIC     - Tweaking axis range, or labels.
# MAGIC     - Reordering or recoloring series for clarity.
# MAGIC     - Turning on data labels or changing formatting for precise value display.
# MAGIC
# MAGIC **Step 8: Save or Download**  
# MAGIC   + When satisfied, click **Save** in the bottom-right corner to store the chart in your Notebook.
# MAGIC   + Download ⤓ options are provided if you want to use the chart in presentations or share it in reports.

# COMMAND ----------

display(telco_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploring Data with Custom Aggregation
# MAGIC
# MAGIC Taking a closer look at `PaymentMethod`, let's review the distribution of the data for this column by calculating aggregated counts per payment type.  
# MAGIC
# MAGIC In this section, we utilize the **`groupBy`** and **`count`** functions to analyze the distribution of payment methods in the Telco Churn dataset. The `orderBy` clause is applied to present the results in an organized manner, providing insights into the frequency of each payment method within the dataset.

# COMMAND ----------

display(telco_df.groupBy("PaymentMethod").count().orderBy("count", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Observation: Identifying Payment Method Distribution**
# MAGIC
# MAGIC Through the groupBy command on the "`PaymentMethod`" column, we observe the distribution of payment methods within the Telco Churn dataset. The result reveals that **Electronic check** is the most frequently used method, while other methods such as Bank transfer and Credit card (automatic) also exhibit substantial usage. 
# MAGIC
# MAGIC Additionally, there are **704 instances with missing payment method** information.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### External Visualization Tools
# MAGIC
# MAGIC You can use any external visualization library to create custom visualization In this demo, we will use popular python library `seaborn` and `matplotlib` for creating custom visualizations.
# MAGIC
# MAGIC **First, we will convert the DataFrame to Pandas.**

# COMMAND ----------

# Convert to pandas dataframe 
telco_pdf = telco_df.toPandas()

# COMMAND ----------

display(telco_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### 1. Correlation Heatmap
# MAGIC
# MAGIC We will import the **`seaborn`** library, to create a correlation heatmap to visually represent the correlation matrix among numerical features. This provides insights into the strength and direction of relationships between variables.

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Select columns that are the numerical columns
selected_columns = ['tenure', 'TotalCharges', 'MonthlyCharges']

# Select the specified columns from the DataFrame
telco_corr = telco_pdf[selected_columns].corr()

# Create a heatmap using seaborn
plt.figure(figsize=(10, 6))
sns.heatmap(telco_corr, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap for Telco Dataset')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Pairplot for Numerical Variables:
# MAGIC
# MAGIC Generate a pairplot to visualize relationships between **numerical variables**. This provides a quick overview of how features interact and whether certain patterns emerge based on the `'Churn'` status.
# MAGIC

# COMMAND ----------

# Select columns that are the numerical columns
selected_columns = ['tenure', 'TotalCharges', 'MonthlyCharges']

# Select the specified columns from the DataFrame
telco_ppdf = telco_pdf[selected_columns + ['Churn']]

# Pairplot for a quick overview of relationships between numerical variables
sns.pairplot(telco_ppdf, hue='Churn', diag_kind='kde')
plt.suptitle('Pairplot for Telco Dataset', y=1.02)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### 3. Boxplot for Monthly Charges by Churn:
# MAGIC
# MAGIC Create a boxplot to visually compare the distribution of monthly charges between customers who churned and those who didn't.

# COMMAND ----------

# Boxplot for visualizing the distribution of Monthly Charges by Churn
plt.figure(figsize=(10, 6))
sns.boxplot(x='Churn', y='MonthlyCharges', data=telco_pdf)
plt.title('Monthly Charges Distribution by Churn Status')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Write dataframe to delta (bronze) table

# COMMAND ----------

# Specify the desired name for the table
table_name_bronze = "telco_missing_bronze"
telco_df.write.saveAsTable(table_name_bronze) # will be stored under default catalog and schema

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Time-Travel with Delta
# MAGIC
# MAGIC Now, let's explore the fascinating concept of time-travel with Delta. Here, we're delving into versioning, allowing us to read a specific or previous version of our dataset. It's a powerful feature, but in the realm of machine learning, caution is advised. 
# MAGIC
# MAGIC **💡 Note:** While versioning can be crucial for reproducibility, **it may pose challenges for ML, where experiments often span longer timeframes than typical data retention periods**. The delicate balance between versioning and ML practices is a topic to tread carefully, keeping in mind potential drawbacks when utilizing previous dataset versions.

# COMMAND ----------

# Drop columns and overwrite table
to_drop_wrong = ["gender", "SeniorCitizen"]
telco_dropped_df = telco_df.drop(*to_drop_wrong)

telco_dropped_df.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(table_name_bronze)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reverting Changes by Version
# MAGIC
# MAGIC What to do, if we drop a column by mistake?
# MAGIC
# MAGIC With Delta's powerful time-travel feature, we can seamlessly revert to a previous version of our dataset. Let's initiate this process by using the **'`DESCRIBE HISTORY`'** SQL command. This command provides us with a comprehensive history of changes, allowing us to pinpoint the version where '`SeniorCitizen`' was still part of our dataset. 
# MAGIC
# MAGIC Let's explore and revert our unintentional omission.

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY telco_missing_bronze

# COMMAND ----------

# MAGIC %md
# MAGIC Examine the schema to validate whether columns were dropped, specifically verifying if the 'gender' & 'SeniorCitizen' columns have been removed.

# COMMAND ----------

spark.table(table_name_bronze).printSchema()

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE `telco_missing_bronze`;

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Reverting Changes by Version
# MAGIC  
# MAGIC As we can see in the **`operation`** column in version 1, the table was overwritten, leading to the unintentional removal of columns. To rectify this, we must now perform a time-travel operation, reverting to version 0, to retrieve the table with all the original columns intact.

# COMMAND ----------

telco_df_v0 = (
  spark.read
      .option("versionAsOf", 0)
      .table(table_name_bronze)
)

telco_df_v0.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Reverting Changes by Timestamp
# MAGIC
# MAGIC You can also query based upon **`timestamp`**.  
# MAGIC
# MAGIC **Note that the ability to query an older snapshot of a table (time travel) is lost after running <a href="https://docs.databricks.com/en/sql/language-manual/delta-vacuum.html" target="_blank">a VACUUM command.</a>**

# COMMAND ----------

# Extract timestamp of first version (can also be set manually)
timestamp_v0 = spark.sql(f"DESCRIBE HISTORY telco_missing_bronze").orderBy("version").first().timestamp
print(timestamp_v0)

# COMMAND ----------

(spark
        .read
        .option("timestampAsOf", timestamp_v0)
        .table("telco_missing_bronze")
        .printSchema()
)

# COMMAND ----------

# MAGIC %md
# MAGIC Based on original version we can specifically drop the correct columns of disinterest and save the table again

# COMMAND ----------

to_drop = ['']
telco_dropped_df = telco_df_v0.drop(*to_drop)

telco_dropped_df.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(table_name_bronze)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Version 2 is our latest and most accurate table version.

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY `telco_missing_bronze`;

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC This demo successfully navigated the essential aspects of data management and exploration using Delta tables and Pandas. We seamlessly read and wrote data, performed time-travel operations for versioning, and computed insightful statistics for machine learning. The integrated visualizations, including correlation matrices and pair plots, provided a deeper understanding of the Telco dataset. By showcasing the power of Delta and Pandas, we've equipped ourselves with valuable tools to efficiently manage, analyze, and derive meaningful insights from our data.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="blank">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy" target="blank">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use" target="blank">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/" target="blank">Support</a>