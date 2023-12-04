# Databricks notebook source
from IPython.display import Image


# COMMAND ----------

# MAGIC %fs ls dbfs:/ml/blogs/gan/dpm_xl/evographnet/data-extended/

# COMMAND ----------

# MAGIC %sh cat /dbfs/ml/blogs/gan/dpm_xl/evographnet/data-extended/cortical.lh.ShapeConnectivityTensor_OAS2_00278_MR1_t1.txt

# COMMAND ----------

# MAGIC %sh cat /dbfs/ml/blogs/gan/dpm_xl/evographnet/data-extended/create_data.py

# COMMAND ----------

# MAGIC %fs ls dbfs:/ml/blogs/gan/dpm_xl/evographnet/losses

# COMMAND ----------

import pickle

# Path to your pickle file on DBFS or mounted storage
file_path = "/dbfs/ml/blogs/gan/dpm_xl/evographnet/losses/GAN2_ValLoss_exp_0"

# Load the contents of the file
with open(file_path, 'rb') as file:
    gan2_valloss_exp_0_data = pickle.load(file)

display(gan2_valloss_exp_0_data)

# COMMAND ----------

# Here's how to format these metrics in a DataFrame

import os
from pyspark.sql.functions import lit
from pyspark.sql.types import StructType, StructField, FloatType, StringType
import numpy as np

# Extract filename from the file path
gan2_valloss_exp_0_filepath = "/dbfs/ml/blogs/gan/dpm_xl/evographnet/losses/GAN2_ValLoss_exp_0"
gan2_valloss_exp_0_filename = os.path.basename(gan2_valloss_exp_0_filepath)

gan2_valloss_exp_0_data_list = list(map(tuple, gan2_valloss_exp_0_data.tolist()))

schema = StructType([StructField(f"epoch_{i}", FloatType(), True) for i in range(len(gan2_valloss_exp_0_data_list[0]))])

gan2_valloss_exp_0_df = spark.createDataFrame(gan2_valloss_exp_0_data_list, schema)

display(gan2_valloss_exp_0_df)

# gan2_valloss_exp_0_df.write.format("delta").saveAsTable("")

# COMMAND ----------

import os
from pyspark.sql.functions import lit
# Extract filename from the file path
gan2_valloss_exp_0_filepath = "/dbfs/ml/blogs/gan/dpm_xl/evographnet/losses/GAN2_ValLoss_exp_0"
gan2_valloss_exp_0_filename = os.path.basename(gan2_valloss_exp_0_filepath)

# Add filename column
gan2_valloss_exp_0_df_with_filename = gan2_valloss_exp_0_df.withColumn("filepath", lit(gan2_valloss_exp_0_filepath)).withColumn("model_stage", lit(gan2_valloss_exp_0_filename))
display(gan2_valloss_exp_0_df_with_filename)


# COMMAND ----------

df_with_filename = df.withColumn("filename", lit(filename))

# COMMAND ----------

# MAGIC %sh cat /dbfs/ml/blogs/gan/dpm_xl/evographnet/losses/GAN2_ValLoss_exp_0

# COMMAND ----------

# MAGIC %fs ls dbfs:/ml/blogs/gan/dpm_xl/evographnet/weights

# COMMAND ----------

import torch
generator_2_499_0_model = torch.load('/dbfs/ml/blogs/gan/dpm_xl/evographnet/weights/generator_2_499_0')
type(generator_2_499_0_model)

# COMMAND ----------

# MAGIC %sh cat /dbfs/ml/blogs/gan/dpm_xl/evographnet/weights/generator_2_499_0

# COMMAND ----------

# MAGIC %fs ls dbfs:/ml/blogs/gan/dpm_xl/evographnet/plots

# COMMAND ----------

display(Image(filename="/dbfs/ml/blogs/gan/dpm_xl/evographnet/plots/TP_Loss_2_ValSet0_exp0.png"))

# COMMAND ----------

display(Image(filename="/dbfs/ml/blogs/gan/dpm_xl/evographnet/plots/TP_Loss_2_ValSet1_exp0.png"))

# COMMAND ----------

display(Image(filename="/dbfs/ml/blogs/gan/dpm_xl/evographnet/plots/TP_Loss_2_ValSet2_exp0.png"))

# COMMAND ----------

# MAGIC %md ### Run inference on this PyTorch GAN model
# MAGIC

# COMMAND ----------

import torch
generator_2_499_0_model = torch.load('/dbfs/ml/blogs/gan/dpm_xl/evographnet/weights/generator_2_499_0')
type(generator_2_499_0_model)

# COMMAND ----------


