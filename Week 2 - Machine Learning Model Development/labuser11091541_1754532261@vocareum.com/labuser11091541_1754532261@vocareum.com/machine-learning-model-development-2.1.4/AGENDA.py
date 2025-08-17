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
# MAGIC ## Machine Learning Model Development
# MAGIC
# MAGIC This comprehensive course provides a practical guide to developing traditional machine learning models on Databricks, emphasizing hands-on demonstrations and workflows using popular ML libraries. This course focuses on executing common tasks efficiently with **`AutoML`** and **`MLflow`**. Participants will delve into key topics, including regression and classification models, harnessing Databricks' capabilities to track model training, leveraging feature stores for model development, and implementing hyperparameter tuning. Additionally, the course covers AutoML for rapid and low-code model training, ensuring that participants gain practical, real-world skills for streamlined and effective machine learning model development in the Databricks environment.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Prerequisites
# MAGIC The content was developed for participants with these skills/knowledge/abilities:  
# MAGIC - Knowledge of fundamental concepts of regression and classification methods
# MAGIC - Familiarity with Databricks workspace and notebooks
# MAGIC - Intermediate level knowledge of Python
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Course Agenda  
# MAGIC The following modules are part of the **Machine Learning Model Development** course by **Databricks Academy**.
# MAGIC
# MAGIC | # | Module Name | Lesson Name |
# MAGIC |---|-------------|-------------|
# MAGIC | 1 | **[Model Development Workflow]($./M01 - Model Development Workflow)** | • *Lecture:* Model Development Workflow <br> • [**Demo:** Supervised Learning]($./M01 - Model Development Workflow/1.1a Demo - Supervised Learning) <br> • [**Demo:** Unsupervised Learning]($./M01 - Model Development Workflow/1.1b Demo - Unsupervised Learning) <br> • [**Demo:** Model Tracking with MLflow]($./M01 - Model Development Workflow/1.2 Demo - Model Tracking with MLflow) <br> • [**Lab:** Model Development Tracking with MLflow]($./M01 - Model Development Workflow/1.3 Lab - Model Development Tracking with MLflow) |
# MAGIC | 2 | **[Hyperparameter Tuning]($./M02 - Hyperparameter Tuning)** | • *Lecture:* Hyperparameter Tuning <br> • [**Demo:** Hyperparameter Tuning with Optuna]($./M02 - Hyperparameter Tuning/2.1 Demo - Hyperparameter Tuning with Optuna) <br> • [**Lab:** Hyperparameter Tuning with Optuna]($./M02 - Hyperparameter Tuning/2.2 Lab - Hyperparameter Tuning with Optuna) |
# MAGIC | 3 | **[AutoML]($./M03 - AutoML)** | • *Lecture:* AutoML <br> • [**Demo:** Automated Model Development with AutoML]($./M03 - AutoML/3.1 Demo - Automated Model Development with AutoML) <br> • [**Lab:** AutoML]($./M03 - AutoML/3.2 Lab - AutoML) |
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