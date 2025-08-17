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
# MAGIC ## Machine Learning Model Deployment
# MAGIC
# MAGIC This course is designed to introduce three primary machine learning deployment strategies and illustrate the implementation of each strategy on Databricks. Following an exploration of the fundamentals of model deployment, the course delves into batch inference, offering hands-on demonstrations and labs for utilizing a model in batch inference scenarios, along with considerations for performance optimization. The second part of the course comprehensively covers pipeline deployment, while the final segment focuses on real-time deployment. Participants will engage in hands-on demonstrations and labs, deploying models with Model Serving and utilizing the serving endpoint for real-time inference.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Prerequisites
# MAGIC The content was developed for participants with these skills/knowledge/abilities:  
# MAGIC - Knowledge of fundamental machine learning models
# MAGIC - Knowledge of model lifecycle and MLflow components
# MAGIC - Familiarity with Databricks workspace and notebooks
# MAGIC - Intermediate level knowledge of Python
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Course Agenda  
# MAGIC The following modules are part of the **Machine Learning Model Deployment** course by **Databricks Academy**.
# MAGIC
# MAGIC | # | Module Name | Lesson Name |
# MAGIC |---|-------------|-------------|
# MAGIC | 1 | **Model Deployment Fundamentals** | • *Lecture:* Model Deployment Strategies <br> • *Lecture:* Model Deployment with MLflow |
# MAGIC | 2 | **[Batch Deployment]($./M02 - Batch Deployment)** | • *Lecture:* Introduction to Batch Deployment <br> • [**Demo:** Batch Deployment]($./M02 - Batch Deployment/2.1 Demo - Batch Deployment) <br> • [**Lab:** Batch Deployment]($./M02 - Batch Deployment/2.2 Lab - Batch Deployment) |
# MAGIC | 3 | **[Pipeline Deployment]($./M03 - Pipeline Deployment)** | • *Lecture:* Introduction to Pipeline Deployment <br> • [**Demo:** Pipeline Deployment]($./M03 - Pipeline Deployment/3.1a Demo - Pipeline Deployment) <br> • [**Demo:** Inference Pipeline]($./M03 - Pipeline Deployment/3.1b Demo - Inference Pipeline) |
# MAGIC | 4 | **[Real-time Deployment and Online Stores]($./M04 - Real-time Deployment and Online Stores)** | • *Lecture:* Introduction to Real-time Deployment <br> • *Lecture:* Databricks Model Serving <br> • [**Demo:** Real-time Deployment with Model Serving]($./M04 - Real-time Deployment and Online Stores/4.1 Demo - Real-time Deployment with Model Serving) <br> • [**Demo:** Custom Model Deployment with Model Serving]($./M04 - Real-time Deployment and Online Stores/4.2 Demo - Custom Model Deployment with Model Serving) <br> • [**Lab:** Real-time Deployment with Model Serving]($./M04 - Real-time Deployment and Online Stores/4.3 Lab - Real-time Deployment with Model Serving) |
# MAGIC
# MAGIC
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