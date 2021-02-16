from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

categoricalColumns = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
stages = [] # stages in our Pipeline

for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + "Index")
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]

label_stringIdx = StringIndexer(inputCol="income", outputCol="label")
stages += [label_stringIdx]
numericCols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

pipeline = Pipeline(stages=stages)

pipelineModel = pipeline.fit(df_adult)

dataset = pipelineModel.transform(df_adult)
selectedcols = ["label", "features"] + cols
dataset = dataset.select(selectedcols)

(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)


from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)

lrModel = lr.fit(trainingData)


lrTrainingSummary = lrModel.summary

  

# make predictions on test data & # display predictions
lrPredictions = lrModel.transform(testData)


from pyspark.ml.tuning import ParamGridBuilder, CrossValidator



# set up an evaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
lrCv = CrossValidator(estimator=lr, estimatorParamMaps=lrParamGrid, evaluator=evaluator, numFolds=2)
lrCvModel = lrCv.fit(trainingData)
lrCvPredictions = lrCvModel.transform(testData)


lrWeights = lrCvModel.bestModel.coefficients
lrWeights = [(float(w),) for w in lrWeights]  # convert numpy type to float, and to tuple
lrWeightsDF = sqlContext.createDataFrame(lrWeights, ["Feature Weight"])
