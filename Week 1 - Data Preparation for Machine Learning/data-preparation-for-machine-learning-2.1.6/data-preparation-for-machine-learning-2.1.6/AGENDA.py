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
# MAGIC
# MAGIC
# MAGIC ## Data Preparation for Machine Learning
# MAGIC
# MAGIC This course focuses on the fundamentals of preparing data for machine learning using Databricks. Participants will learn essential skills for exploring, cleaning, and organizing data tailored for traditional machine learning applications. Key topics include data visualization, feature engineering, and optimal feature storage strategies. Through practical exercises, participants will gain hands-on experience in efficiently preparing data sets for machine learning within the Databricks. This course is designed for associate-level data scientists, machine learning practitioners and individuals seeking to enhance their proficiency in data preparation, ensuring a solid foundation for successful machine learning model deployment.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Prerequisites
# MAGIC The content was developed for participants with these skills/knowledge/abilities:  
# MAGIC - Familiarity with Databricks workspace and notebooks
# MAGIC - Familiarity with Delta Lake and Lakehouse
# MAGIC - Intermediate level knowledge of Python
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Course Agenda  
# MAGIC The following modules are part of the **Data Preparation for Machine Learning** course by **Databricks Academy**.
# MAGIC
# MAGIC | # | Module Name | Lesson Name |
# MAGIC |---|-------------|-------------|
# MAGIC | 1 | [Managing and Exploring Data]($./M01 - Managing and Exploring Data) | • *Lecture:* Introduction <br> • *Lecture:* Managing and Exploring Data in the Lakehouse <br> • [Demo: Load and Explore Data]($./M01 - Managing and Exploring Data/1.1 Demo - Load and Explore Data) <br> • [Lab: Load and Explore Data]($./M01 - Managing and Exploring Data/1.2 Lab - Load and Explore Data) |
# MAGIC | 2 | [Data Preparation and Feature Engineering]($./M02 - Data Preparation and Feature Engineering) | • *Lecture:* Introduction <br> • *Lecture:* Fundamentals of Data Preparation and Feature Engineering <br> • *Lecture:* Data Imputation <br> • *Lecture:* Data Encoding <br> • *Lecture:* Data Standardization <br> • [Demo: Data Imputation and Transformation Pipeline]($./M02 - Data Preparation and Feature Engineering/2.1 Demo - Data Imputation and Transformation Pipeline) <br> • [Demo: Build a Feature Engineering Pipeline with Embeddings]($./M02 - Data Preparation and Feature Engineering/2.2 Demo - Build a Feature Engineering Pipeline with Embeddings) <br> • [Lab: Build a Feature Engineering Pipeline]($./M02 - Data Preparation and Feature Engineering/2.3 Lab - Build a  Feature Engineering Pipeline) |
# MAGIC | 3 | [Feature Store]($./M03 - Feature Store) | • *Lecture:* Introduction to Feature Store <br> • [Demo: Using Feature Store for Feature Engineering]($./M03 - Feature Store/3.1 Demo - Using Feature Store for Feature Engineering) <br> • [Lab: Feature Engineering with Feature Store]($./M03 - Feature Store/3.2 Lab - Feature Engineering with Feature Store) |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * Use Databricks Runtime version: **`16.3.x-cpu-ml-scala2.12`** for running all demo and lab notebooks.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2025 Databricks, Inc. All rights reserved. Apache, Apache Spark, Spark, the Spark Logo, Apache Iceberg, Iceberg, and the Apache Iceberg logo are trademarks of the <a href="https://www.apache.org/" target="blank">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy" target="blank">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use" target="blank">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/" target="blank">Support</a>