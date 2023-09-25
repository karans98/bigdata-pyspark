
# Importing the required packages 
from pyspark.ml.classification import (LogisticRegression, RandomForestClassifier, NaiveBayes)
from pyspark.ml.evaluation import (BinaryClassificationEvaluator, MulticlassClassificationEvaluator)
from pyspark.ml import Pipeline
from pyspark.ml.feature import (VectorAssembler, StringIndexer) 
from pyspark.ml.stat import Correlation
import pyspark.sql.functions as x
from pyspark.sql.functions import col, count, isnan, when
from pyspark_dist_explore import hist

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# installing package to plot histogram
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()

#Creating spark context-Its like connecting to spark cluster
from pyspark import SparkConf 
from pyspark.context import SparkContext 
sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

#importing the Data File into Google Colab
from google.colab import files
files.upload()

# Loading the csv file
df = spark.read.csv('Raisin_Dataset.csv',inferSchema=True, header =True)

# Printing the dataframe Schema, data types
df.printSchema()

# Printing the top 10 rows of the dataFrame
df.show(10)

#Summary statistics on every column
print("Summary Statistics on every column of dataframe")
describe_stats = df.describe()
print(describe_stats.show())

# Checking for null values
df.select([x.count(x.when(x.isnull(c), c)).alias(c) for c in df.columns]).show()

fig, ax = plt.subplots()
hist(ax,df.select("Area"), bins = 30, color=['blue'])

# Plotting the pairplot
print("Pairwise Plots")
sns.pairplot(df.toPandas())

#Target variable values
target_counts = df.groupby('Class').count()
print("Target value counts of each category:")
target_counts.show()

#Plotting the Box plots
print("boxplots of numeric columns")
df1 = df.drop('Class')
for i in df1.columns:
  l = df1.select(i).collect()
  l = [r[0] for r in l]
  plt.figure()
  b = sns.boxplot(x = l)
  plt.title(i)
  plt.show()

#Generating the Corelation Matrix 
columnsData=df1.columns
vector_col = "corr_features"
assembler = VectorAssembler(inputCols=columnsData, outputCol=vector_col)
myGraph_vector = assembler.transform(df1)
matrix1 = Correlation.corr(myGraph_vector, vector_col).collect()[0][0]
corrmatrix = matrix1.toArray().tolist()
print("The generated Correlation Matrix is:: \n {0}".format(corrmatrix))

# Generating the HeatMap of the Data
figure=plt.figure(1)
figure1=figure.add_subplot(111)
figure1.set_xticklabels(['']+columnsData)
figure1.set_yticklabels(['']+columnsData)
figure.colorbar(figure1.matshow(corrmatrix,vmax=1,vmin=-1))
plt.show()

# Dropping highly correlated columns
cols = ["MajorAxisLength", "MinorAxisLength", "ConvexArea", "Perimeter"]
df = df.drop(*cols)

df.printSchema()

#splitting data into train and test sets
train_df, test_df = df.randomSplit([.7,.3])

label = 'Class'
numerical_cols = ['Area', 'Eccentricity','Extent']

# Indexer for classification label:
label_indexer = StringIndexer(inputCol=label, outputCol=label+'_indexed')

#assemble all features as vector to be used as input for Spark MLLib
assembler = VectorAssembler(inputCols= numerical_cols, outputCol='features')

# Creating data processing pipeline
pipeline = Pipeline(stages= [label_indexer, assembler])

lr = LogisticRegression(featuresCol='features', labelCol=label+'_indexed')
rfc = RandomForestClassifier(featuresCol='features', labelCol=label+'_indexed', numTrees=100)
nb = NaiveBayes(featuresCol='features', labelCol=label+'_indexed')

# creating pipelines with machine learning models
pipeline_lr = Pipeline(stages=[pipeline, lr])
pipeline_rfc = Pipeline(stages=[pipeline, rfc])
pipeline_nb = Pipeline(stages=[pipeline, nb])

#fitting models with train subset
lr_fit = pipeline_lr.fit(train_df)
rfc_fit = pipeline_rfc.fit(train_df)
nb_fit = pipeline_nb.fit(train_df)

# predictions for test subset
pred_lr = lr_fit.transform(test_df)
pred_rfc = rfc_fit.transform(test_df)
pred_nb = nb_fit.transform(test_df)

pred_AUC = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol=label+'_indexed')

AUC_lr = pred_AUC.evaluate(pred_lr)
AUC_rfc = pred_AUC.evaluate(pred_rfc)
AUC_nb = pred_AUC.evaluate(pred_nb)
print("The Area Under the ROC Curve for Logistic Regression, Random Forest ")
print(AUC_lr, AUC_rfc, AUC_nb)
print('Logistic Regression AUC: ', '{:.2f}'.format(AUC_lr))
print('Random Forest AUC: ', '{:.2f}'.format(AUC_rfc))
print('Naive Bayes AUC: ', '{:.2f}'.format(AUC_nb))

acc_evaluator = MulticlassClassificationEvaluator(labelCol=label+'_indexed', predictionCol="prediction", metricName="accuracy")

acc_lr = acc_evaluator.evaluate(pred_lr)
acc_rfc = acc_evaluator.evaluate(pred_rfc)
acc_nb = acc_evaluator.evaluate(pred_nb)
 
print('Logistic Regression accuracy: ', '{:.2f}'.format(acc_lr*100), '%', sep='')
print('Random Forest accuracy: ', '{:.2f}'.format(acc_rfc*100), '%', sep='')
print('Naive Bayes accuracy: ', '{:.2f}'.format(acc_nb*100), '%', sep='')

