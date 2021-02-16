# Databricks notebook source
# MAGIC %md
# MAGIC ## Exercise Overview
# MAGIC In this exercise we will play with Spark [Datasets & Dataframes](https://spark.apache.org/docs/latest/sql-programming-guide.html#datasets-and-dataframes), some [Spark SQL](https://spark.apache.org/docs/latest/sql-programming-guide.html#sql), and build a couple of binary classifiaction models using [Spark ML](https://spark.apache.org/docs/latest/ml-guide.html) (with some [MLlib](https://spark.apache.org/mllib/) too). 
# MAGIC <br><br>
# MAGIC The set up and approach will not be too dissimilar to the standard type of approach you might do in [Sklearn](http://scikit-learn.org/stable/index.html). Spark has matured to the stage now where for 90% of what you need to do (when analysing tabular data) should be possible with Spark dataframes, SQL, and ML libraries. This is where this exercise is mainly trying to focus.  
# MAGIC <br>
# MAGIC Feel free to adapt this exercise to play with other datasets readily availabe in the Databricks enviornment (they are listed in a cell below). 
# MAGIC 
# MAGIC ##### Getting Started
# MAGIC To get started you will need to create and attach a databricks spark cluster to this notebook. This notebook was developed on a cluster created with: 
# MAGIC - Databricks Runtime Version 4.0 (includes Apache Spark 2.3.0, Scala 2.11)
# MAGIC - Python Version 3
# MAGIC 
# MAGIC ##### Links & References
# MAGIC 
# MAGIC Some useful links and references of sources used in creating this exercise:
# MAGIC 
# MAGIC **Note**: Right click and open as new tab!
# MAGIC <br>
# MAGIC 1. [Latest Spark Docs](https://spark.apache.org/docs/latest/index.html)
# MAGIC 1. [Databricks Homepage](https://databricks.com/)
# MAGIC 1. [Databricks Community Edition FAQ](https://databricks.com/product/faq/community-edition)
# MAGIC 1. [Databricks Self Paced Training](https://databricks.com/training-overview/training-self-paced)
# MAGIC 1. [Databricks Notebook Guide](https://docs.databricks.com/user-guide/notebooks/index.html)
# MAGIC 1. [Databricks Binary Classification Tutorial](https://docs.databricks.com/spark/latest/mllib/binary-classification-mllib-pipelines.html#binary-classification)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Get Data
# MAGIC 
# MAGIC Here we will pull in some sample data that is already pre-loaded onto all databricks clusters.
# MAGIC 
# MAGIC Feel free to adapt this notebook later to play around with a different dataset if you like (all available are listed in a cell below).

# COMMAND ----------

# display datasets already in databricks
display(dbutils.fs.ls("/databricks-datasets"))

# COMMAND ----------

# MAGIC %md
# MAGIC Lets take a look at the '**adult**' dataset on the filesystem. This is the typical US Census data you often see online in tutorials. [Here](https://archive.ics.uci.edu/ml/datasets/adult) is the same data in the UCI repository.
# MAGIC 
# MAGIC _As an aside: [here](https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/census) this same dataset is used as a quickstart example for Google CLoud ML & Tensorflow Estimator API (in case youd be interested in playing with tensorflow on the same dataset as here)._

# COMMAND ----------

# MAGIC %fs ls databricks-datasets/adult/adult.data

# COMMAND ----------

# MAGIC %md
# MAGIC **Note**: Above  %fs is just some file system cell magic that is specific to databricks. More info [here](https://docs.databricks.com/user-guide/notebooks/index.html#mix-languages).

# COMMAND ----------

# MAGIC %md
# MAGIC #### Spark SQL
# MAGIC 
# MAGIC Below we will use Spark SQL to load in the data and then register it as a Dataframe aswell. So the end result will be a Spark SQL table called _adult_ and a Spark Dataframe called _df_adult_. 
# MAGIC <br><br>
# MAGIC This is an example of the flexibility in Spark in that you could do lots of you ETL and data wrangling using either Spark SQL or Dataframes and pyspark. Most of the time it's a case of using whatever you are most comfortable with.
# MAGIC <br><br>
# MAGIC When you get more advanced then you might looking the pro's and con's of each and when you might favour one or the other (or operating direclty on RDD's), [here](https://databricks.com/blog/2016/07/14/a-tale-of-three-apache-spark-apis-rdds-dataframes-and-datasets.html) is a good article on the issues. For now, no need to overthink it!

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- drop the table if it already exists
# MAGIC DROP TABLE IF EXISTS adult

# COMMAND ----------

# MAGIC %sql
# MAGIC -- create a new table in Spark SQL from the datasets already loaded in the underlying filesystem.
# MAGIC -- In the real world you might be pointing at a file on HDFS or a hive table etc. 
# MAGIC CREATE TABLE adult (
# MAGIC   age DOUBLE,
# MAGIC   workclass STRING,
# MAGIC   fnlwgt DOUBLE,
# MAGIC   education STRING,
# MAGIC   education_num DOUBLE,
# MAGIC   marital_status STRING,
# MAGIC   occupation STRING,
# MAGIC   relationship STRING,
# MAGIC   race STRING,
# MAGIC   sex STRING,
# MAGIC   capital_gain DOUBLE,
# MAGIC   capital_loss DOUBLE,
# MAGIC   hours_per_week DOUBLE,
# MAGIC   native_country STRING,
# MAGIC   income STRING)
# MAGIC USING com.databricks.spark.csv
# MAGIC OPTIONS (path "/databricks-datasets/adult/adult.data", header "true")

# COMMAND ----------

# look at the data
#spark.sql("SELECT * FROM adult LIMIT 5").show() 
# this will look prettier in Databricks if you use display() instead
display(spark.sql("SELECT * FROM adult LIMIT 5"))

# COMMAND ----------

# MAGIC %md
# MAGIC If you are more comfortable with SQL then as you can see below, its very easy to just get going with writing standard SQL type code to analyse your data, do data wrangling and create new dataframes.

# COMMAND ----------

# Lets get some summary marital status rates by occupation
result = spark.sql(
  """
  SELECT 
    occupation,
    SUM(1) as n,
    ROUND(AVG(if(LTRIM(marital_status) LIKE 'Married-%',1,0)),2) as married_rate,
    ROUND(AVG(if(lower(marital_status) LIKE '%widow%',1,0)),2) as widow_rate,
    ROUND(AVG(if(LTRIM(marital_status) = 'Divorced',1,0)),2) as divorce_rate,
    ROUND(AVG(if(LTRIM(marital_status) = 'Separated',1,0)),2) as separated_rate,
    ROUND(AVG(if(LTRIM(marital_status) = 'Never-married',1,0)),2) as bachelor_rate
  FROM 
    adult 
  GROUP BY 1
  ORDER BY n DESC
  """)
display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC You can easily register dataframes as a table for Spark SQL too. So this way you can easily move between Dataframes and Spark SQL for whatever reason.

# COMMAND ----------

# register the df we just made as a table for spark sql
sqlContext.registerDataFrameAsTable(result, "result")
spark.sql("SELECT * FROM result").show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC #### <span style="color:darkblue">Question 1</span>
# MAGIC 
# MAGIC 1. Write some spark sql to get the top 'bachelor_rate' by 'education' group?

# COMMAND ----------

### Question 1.1 Answer ###

result = spark.sql(
  """
  SELECT 
    education,
    SUM(1) as n,
    ROUND(AVG(if(LTRIM(marital_status) LIKE 'Married-%',1,0)),2) as married_rate,
    ROUND(AVG(if(lower(marital_status) LIKE '%widow%',1,0)),2) as widow_rate,
    ROUND(AVG(if(LTRIM(marital_status) = 'Divorced',1,0)),2) as divorce_rate,
    ROUND(AVG(if(LTRIM(marital_status) = 'Separated',1,0)),2) as separated_rate,
    ROUND(AVG(if(LTRIM(marital_status) = 'Never-married',1,0)),2) as bachelor_rate
  FROM 
    adult 
  GROUP BY 1
  ORDER BY n DESC
  """) # fill in here
result.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Spark DataFrames
# MAGIC 
# MAGIC Below we will create our DataFrame from the SQL table and do some similar analysis as we did with Spark SQL but using the DataFrames API.

# COMMAND ----------

# register a df from the sql df
df_adult = spark.table("adult")
cols = df_adult.columns # this will be used much later in the notebook, ignore for now

# COMMAND ----------

# look at df schema
df_adult.printSchema()

# COMMAND ----------

# look at the df
display(df_adult)
#df_adult.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC Below we will do a similar calculation to what we did above but using the DataFrames API

# COMMAND ----------

# import what we will need
from pyspark.sql.functions import when, col, mean, desc, round

# wrangle the data a bit
df_result = df_adult.select(
  df_adult['occupation'], 
  # create a 1/0 type col on the fly
  when( col('marital_status') == 'Unmarried' , 1 ).otherwise(0).alias('is_divorced')
)
# do grouping (and a round)
df_result = df_result.groupBy('occupation').agg(round(mean('is_divorced'),2).alias('divorced_rate'))
# do ordering
df_result = df_result.orderBy(desc('divorced_rate'))
# show results
df_result.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC As you can see the dataframes api is a bit more verbose then just expressing what you want to do in standard SQL.<br><br>But some prefer it and might be more used to it, and there could be cases where expressing what you need to do might just be better using the DataFrame API if it is too complicated for a simple SQL expression for example of maybe involves recursion of some type.

# COMMAND ----------

# MAGIC %md
# MAGIC #### <span style="color:darkblue">Question 2</span>
# MAGIC 1. Write some pyspark to get the top 'bachelor_rate' by 'education' group using DataFrame operations?

# COMMAND ----------

### Question 2.1 Answer ###

# wrangle the data a bit
df_result = df_adult.select(
  df_adult['education'], 
  # create a 1/0 type col on the fly
  when( col('marital_status') == 'Never-married' , 1 ).otherwise(0).alias('is_bachelor')
)
# do grouping (and a round)
df_result = df_result.groupBy('education').agg(round(mean('is_bachelor'),2).alias('bachelor_rate'))
# do ordering
df_result = df_result.orderBy(desc('bachelor_rate')) # fill in here

df_result.show(1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Explore & Visualize Data
# MAGIC 
# MAGIC It's very easy to [collect()](https://spark.apache.org/docs/latest/rdd-programming-guide.html#printing-elements-of-an-rdd) your Spark DataFrame data into a Pandas df and then continue to analyse or plot as you might normally.
# MAGIC <br><br>
# MAGIC Obviously if you try to collect() a huge DataFrame then you will run into issues, so usually you would only collect aggregated or sampled data into a Pandas df.

# COMMAND ----------

import pandas as pd

# do some analysis
result = spark.sql(
  """
  SELECT 
    occupation,
    AVG(IF(income = ' >50K',1,0)) as plus_50k
  FROM 
    adult 
  GROUP BY 1
  ORDER BY 2 DESC
  """)

# collect results into a pandas df
df_pandas = pd.DataFrame(
  result.collect(),
  columns=result.schema.names
)

# look at df
print(df_pandas.head())

# COMMAND ----------

print(df_pandas.describe())

# COMMAND ----------

print(df_pandas.info())

# COMMAND ----------

# MAGIC %md
# MAGIC Here we will just do some very basic plotting to show how you might collect what you are interested in into a Pandas DF and then just plot any way you normally would.
# MAGIC 
# MAGIC For simplicity we are going to use the plotting functionality built into pandas (you could make this a pretty as you want).

# COMMAND ----------

import matplotlib.pyplot as plt

# i like ggplot style
plt.style.use('ggplot')

# get simple plot on the pandas data
myplot = df_pandas.plot(kind='barh', x='occupation', y='plus_50k')

# display the plot (note - display() is a databricks function - 
# more info on plotting in Databricks is here: https://docs.databricks.com/user-guide/visualizations/matplotlib-and-ggplot.html)
display(myplot.figure)

# COMMAND ----------

# MAGIC %md
# MAGIC You can also easily get summary stats on a Spark DataFrame like below. [Here](https://databricks.com/blog/2015/06/02/statistical-and-mathematical-functions-with-dataframes-in-spark.html) is a nice blog post that has more examples.<br><br>So this is an example of why you might want to move from Spark SQL into DataFrames API as being able to just call describe() on the Spark DF is easier then trying to do the equivilant in Spark SQL.

# COMMAND ----------

# describe df
df_adult.select(df_adult['age'],df_adult['education_num']).describe().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### ML Pipeline - Logistic Regression vs Random Forest
# MAGIC 
# MAGIC Below we will create two [Spark ML Pipelines](https://spark.apache.org/docs/latest/ml-pipeline.html) - one that fits a logistic regression and one that fits a random forest. We will then compare the performance of each.
# MAGIC 
# MAGIC **Note**: A lot of the code below is adapted from [this example](https://docs.databricks.com/spark/latest/mllib/binary-classification-mllib-pipelines.html).

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

categoricalColumns = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
stages = [] # stages in our Pipeline

for categoricalCol in categoricalColumns:
    # Category Indexing with StringIndexer
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
    # Use OneHotEncoder to convert categorical variables into binary SparseVectors
    # encoder = OneHotEncoderEstimator(inputCol=categoricalCol + "Index", outputCol=categoricalCol + "classVec")
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    # Add stages.  These are not run here, but will run all at once later on.
    stages += [stringIndexer, encoder]

# COMMAND ----------

# Convert label into label indices using the StringIndexer
label_stringIdx = StringIndexer(inputCol="income", outputCol="label")
stages += [label_stringIdx]

# COMMAND ----------

# Transform all features into a vector using VectorAssembler
numericCols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

# COMMAND ----------

# Create a Pipeline.
pipeline = Pipeline(stages=stages)
# Run the feature transformations.
#  - fit() computes feature statistics as needed.
#  - transform() actually transforms the features.
pipelineModel = pipeline.fit(df_adult)
dataset = pipelineModel.transform(df_adult)
# Keep relevant columns
selectedcols = ["label", "features"] + cols
dataset = dataset.select(selectedcols)
display(dataset)

# COMMAND ----------

### Randomly split data into training and test sets. set seed for reproducibility
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)
print(trainingData.count())
print(testData.count())

# COMMAND ----------

from pyspark.sql.functions import avg

# get the rate of the positive outcome from the training data to use as a threshold in the model
training_data_positive_rate = trainingData.select(avg(trainingData['label'])).collect()[0][0] 

print("Positive rate in the training data is {}".format(training_data_positive_rate))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Logistic Regression - Train

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# Create initial LogisticRegression model
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)

# set threshold for the probability above which to predict a 1
lr.setThreshold(training_data_positive_rate)
# lr.setThreshold(0.5) # could use this if knew you had balanced data

# Train model with Training Data
lrModel = lr.fit(trainingData)

# get training summary used for eval metrics and other params
lrTrainingSummary = lrModel.summary

# Find the best model threshold if you would like to use that instead of the empirical positve rate
fMeasure = lrTrainingSummary.fMeasureByThreshold
maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
lrBestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']) \
    .select('threshold').head()['threshold']
  
print("Best threshold based on model performance on training data is {}".format(lrBestThreshold))

# COMMAND ----------

# MAGIC %md
# MAGIC #### GBM - Train

# COMMAND ----------

# MAGIC %md
# MAGIC #### <span style="color:darkblue">Question 3</span>
# MAGIC 1. Train a GBTClassifier on the training data, call the trained model 'gbModel'

# COMMAND ----------

### Question 3.1 Answer ###
from pyspark.ml.classification import GBTClassifier



# Create initial LogisticRegression model
gb = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10) # fill in here

# Train model with Training Data
gbModel = gb.fit(trainingData) # fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC #### Logistic Regression - Predict

# COMMAND ----------

# make predictions on test data
lrPredictions = lrModel.transform(testData)

# display predictions
display(lrPredictions.select("label", "prediction", "probability"))
#display(lrPredictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ### GBM - Predict

# COMMAND ----------

# MAGIC %md
# MAGIC #### <span style="color:darkblue">Question 4</span>
# MAGIC 1. Get predictions on the test data for your GBTClassifier. Call the predictions df 'gbPredictions'.

# COMMAND ----------

### Question 4.1 Answer ###

# make predictions on test data
gbPredictions = gbModel.transform(testData) # fill in here

display(gbPredictions)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Logistic Regression - Evaluate

# COMMAND ----------

# MAGIC %md
# MAGIC #### <span style="color:darkblue">Question 5</span>
# MAGIC 
# MAGIC 1. Complete the print_performance_metrics() function below to also include measures of F1, Precision, Recall, False Positive Rate and True Positive Rate.

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics

def print_performance_metrics(predictions):
  # Evaluate model
  evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
  auc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
  aupr = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})
  print("auc = {}".format(auc))
  print("aupr = {}".format(aupr))

  # get rdd of predictions and labels for mllib eval metrics
  predictionAndLabels = predictions.select("prediction","label").rdd

  # Instantiate metrics objects
  binary_metrics = BinaryClassificationMetrics(predictionAndLabels)
  multi_metrics = MulticlassMetrics(predictionAndLabels)

  # Area under precision-recall curve
  print("Area under PR = {}".format(binary_metrics.areaUnderPR))
  # Area under ROC curve
  print("Area under ROC = {}".format(binary_metrics.areaUnderROC))
  # Accuracy
  print("Accuracy = {}".format(multi_metrics.accuracy))
  # Confusion Matrix
  print(multi_metrics.confusionMatrix())
  
  ### Question 5.1 Answer ###
  
  # F1 - # fill in here
  print("F1 = {}".format(multi_metrics.fMeasure(1.0)))
  # Precision - # fill in here
  print("Precision = {}".format(multi_metrics.precision(1.0)))
  # Recall - # fill in here
  print("Recall = {}".format(multi_metrics.recall(1.0)))
  # FPR - # fill in here
  print("FPR = {}".format(multi_metrics.falsePositiveRate(1.0)))
  # TPR - # fill in here
  print("TPR = {}".format(multi_metrics.truePositiveRate(1.0)))
  
  
print_performance_metrics(lrPredictions)

# COMMAND ----------

# MAGIC %md
# MAGIC #### GBM - Evaluate

# COMMAND ----------

print_performance_metrics(gbPredictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cross Validation

# COMMAND ----------

# MAGIC %md
# MAGIC For each model you can run the below comand to see its params and a brief explanation of each.

# COMMAND ----------

print(lr.explainParams())

# COMMAND ----------

print(gb.explainParams())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Logisitic Regression - Param Grid

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create ParamGrid for Cross Validation
lrParamGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .addGrid(lr.maxIter, [2, 5])
             .build())

# COMMAND ----------

# MAGIC %md
# MAGIC #### GBM - Param Grid

# COMMAND ----------

# MAGIC %md
# MAGIC #### <span style="color:darkblue">Question 6</span>
# MAGIC 
# MAGIC 1. Build out a param grid for the gb model, call it 'gbParamGrid'.

# COMMAND ----------

### Question 6.1 Answer ###

# Create ParamGrid for Cross Validation
gbParamGrid = (ParamGridBuilder()
             .addGrid(gb.maxDepth, [2, 5])
             .addGrid(gb.maxIter, [10, 100])
             .build())


# fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC #### Logistic Regression - Perform Cross Validation

# COMMAND ----------

# set up an evaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")

# Create CrossValidator
lrCv = CrossValidator(estimator=lr, estimatorParamMaps=lrParamGrid, evaluator=evaluator, numFolds=2)

# Run cross validations
lrCvModel = lrCv.fit(trainingData)

# this will likely take a fair amount of time because of the amount of models that we're creating and testing

# COMMAND ----------

# below approach to getting at the best params from the best cv model taken from:
# https://stackoverflow.com/a/46353730/1919374

# look at best params from the CV
print(lrCvModel.bestModel._java_obj.getRegParam())
print(lrCvModel.bestModel._java_obj.getElasticNetParam())
print(lrCvModel.bestModel._java_obj.getMaxIter())

# COMMAND ----------

# MAGIC %md
# MAGIC #### GBM - Perform Cross Validation

# COMMAND ----------

# MAGIC %md
# MAGIC #### <span style="color:darkblue">Question 7</span>
# MAGIC 1. Perform cross validation of params on your 'gb' model.
# MAGIC 1. Print out the best params you found.

# COMMAND ----------

### Question 7.1 Answer ###

# Create CrossValidator
gbCv = CrossValidator(estimator=gb, estimatorParamMaps=gbParamGrid, evaluator=evaluator, numFolds=2)
# fill in here

# Run cross validations
gbCvModel = gbCv.fit(trainingData)
# fill in here

# COMMAND ----------

### Question 7.2 Answer ###
# fill in here
# look at best params from the CV
print(gbCvModel.bestModel._java_obj.getMaxDepth())
print(gbCvModel.bestModel._java_obj.getMaxIter())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Logistic Regression - CV Model Predict

# COMMAND ----------

# Use test set to measure the accuracy of our model on new data
lrCvPredictions = lrCvModel.transform(testData)

display(lrCvPredictions)

# COMMAND ----------

# MAGIC %md
# MAGIC #### GBM - CV Model Predict

# COMMAND ----------

gbCvPredictions = gbCvModel.transform(testData)

display(gbCvPredictions)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Logistic Regression - CV Model Evaluate

# COMMAND ----------

print_performance_metrics(lrCvPredictions)

# COMMAND ----------

# MAGIC %md
# MAGIC #### GBM - CV Model Evaluate

# COMMAND ----------

print_performance_metrics(gbCvPredictions)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Logistic Regression - Model Explore

# COMMAND ----------

print('Model Intercept: ', lrCvModel.bestModel.intercept)

# COMMAND ----------

lrWeights = lrCvModel.bestModel.coefficients
lrWeights = [(float(w),) for w in lrWeights]  # convert numpy type to float, and to tuple
lrWeightsDF = sqlContext.createDataFrame(lrWeights, ["Feature Weight"])
display(lrWeightsDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Importance

# COMMAND ----------

# MAGIC %md
# MAGIC #### <span style="color:darkblue">Question 8</span>
# MAGIC 1. Print out a table of feature_name and feature_coefficient from the Logistic Regression model.
# MAGIC <br><br>
# MAGIC Hint: Adapt the code from here: https://stackoverflow.com/questions/42935914/how-to-map-features-from-the-output-of-a-vectorassembler-back-to-the-column-name

# COMMAND ----------

### Question 8.1 Answer ###

# from: https://stackoverflow.com/questions/42935914/how-to-map-features-from-the-output-of-a-vectorassembler-back-to-the-column-name

# fill in here


from itertools import chain

attrs = sorted(
    (attr["idx"], attr["name"]) for attr in (chain(*trainingData
        .schema[lrModel.summary.featuresCol]
        .metadata["ml_attr"]["attrs"].values())))


# COMMAND ----------

lrCvFeatureImportance = pd.DataFrame([(name, lrCvModel.bestModel.coefficients [idx]) for idx, name in attrs],columns=['feature_name','feature_importance'])

print(lrCvFeatureImportance.sort_values(by=['feature_importance'],ascending =False))

# COMMAND ----------

gbCvFeatureImportance = pd.DataFrame([(name, gbCvModel.bestModel.featureImportances[idx]) for idx, name in attrs],columns=['feature_name','feature_importance'])

print(gbCvFeatureImportance.sort_values(by=['feature_importance'],ascending =False))

# COMMAND ----------

# MAGIC %md
# MAGIC #### <span style="color:darkblue">Question 9</span>
# MAGIC 1. Build and train a RandomForestClassifier and print out a table of feature importances from it.

# COMMAND ----------

### Question 9.1 Answer ###

from pyspark.ml.classification import RandomForestClassifier # fill in here

rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10) # fill in here

rfModel = rf.fit(trainingData) # fill in here

rfFeatureImportance =  pd.DataFrame([(name, rfModel.featureImportances[idx]) for idx, name in attrs],columns=['feature_name','feature_importance']) # fill in here

print(rfFeatureImportance.sort_values(by=['feature_importance'],ascending =False))

# COMMAND ----------

# DBTITLE 1,Another way to get feature importances


# COMMAND ----------

# Link to code source: 
# https://www.timlrx.com/blog/feature-selection-using-feature-importance-score-creating-a-pyspark-estimator 

def ExtractFeatureImp(featureImp, dataset, featuresCol):
    list_extract = []
    for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))

# COMMAND ----------

# Link to code source: 
# https://www.timlrx.com/blog/feature-selection-using-feature-importance-score-creating-a-pyspark-estimator 

ExtractFeatureImp(rfModel.featureImportances, trainingData, "features")
