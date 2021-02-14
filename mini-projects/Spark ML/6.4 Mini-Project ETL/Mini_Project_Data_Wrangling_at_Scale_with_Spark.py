# Databricks notebook source
# MAGIC %md
# MAGIC # Analyzing Web Server Logs with Apache Spark
# MAGIC 
# MAGIC Apache Spark is an excellent and ideal framework for wrangling, analyzing and modeling on structured and unstructured data - at scale! In this mini-project, we will be focusing on one of the most popular use-cases in the industry - log analytics.
# MAGIC 
# MAGIC Typically, server logs are a very common data source in enterprises and often contain a gold mine of actionable insights and information. Log data comes from many sources in an enterprise, such as the web, client and compute servers, applications, user-generated content, flat files. They can be used for monitoring servers, improving business and customer intelligence, building recommendation systems, fraud detection, and much more.
# MAGIC 
# MAGIC Spark allows you to dump and store your logs in files on disk cheaply, while still providing rich APIs to perform data analysis at scale. This mini-project will show you how to use Apache Spark on real-world production logs from NASA.
# MAGIC You will complete the extract, transform, and load (ETL) process in this Apache Spark enviroment. During this process, you will learn why the ETL process is so crucial to the quality of the machine learning work we will be doing later on.
# MAGIC 
# MAGIC 
# MAGIC There is a total of 15 questions for you to solve along with some interactive examples which will help you learn aspects of leveraging spark for analyzing over 3 million logs at scale.
# MAGIC 
# MAGIC Remember to focus on the __`# TODO: Replace <FILL IN> with appropriate code`__ sections to fill them up with necessary code to solve the desired questions in the notebook

# COMMAND ----------

# MAGIC %md
# MAGIC # Data extraction:

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 1 - Loading up Dependencies

# COMMAND ----------

spark

# COMMAND ----------

sqlContext

# COMMAND ----------

if 'sc' not in locals():
    from pyspark.context import SparkContext
    from pyspark.sql.context import SQLContext
    from pyspark.sql.session import SparkSession
    
    sc = SparkContext()
    sqlContext = SQLContext(sc)
    spark = SparkSession(sc)

# COMMAND ----------

import re
import pandas as pd

# COMMAND ----------

m = re.finditer(r'.*?(spark).*?', "I'm searching for a spark in PySpark", re.I)
for match in m:
    print(match)

# COMMAND ----------

# MAGIC %md
# MAGIC For this mini-project, we will analyze datasets from NASA Kennedy Space Center web server in Florida. The full data set is freely available for download [__here__](http://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html).
# MAGIC 
# MAGIC These two traces contain two month's worth of all HTTP requests to the NASA Kennedy Space Center WWW server in Florida. You can head over to the [__website__](http://ita.ee.lbl.gov/html/contrib/NASA-HTTP.html) and download the following files if needed OR just upload the files we have already provided for you into Domino's Cloud Platform (unless you plan to use Spark locally).
# MAGIC 
# MAGIC - Jul 01 to Jul 31, ASCII format, 20.7 MB gzip compressed, 205.2 MB uncompressed: [ftp://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz](ftp://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz)
# MAGIC - Aug 04 to Aug 31, ASCII format, 21.8 MB gzip compressed, 167.8 MB uncompressed: [ftp://ita.ee.lbl.gov/traces/NASA_access_log_Aug95.gz](ftp://ita.ee.lbl.gov/traces/NASA_access_log_Aug95.gz)
# MAGIC 
# MAGIC Make sure both the data files have been uploaded to Databricks under **"Data" > "DBFS" > "Tables"** as a **.txt** file
# MAGIC 
# MAGIC 
# MAGIC ![DBFS](https://drive.google.com/uc?id=1eE9_CgnUW7psBs_Nlk9qrdD2dXh1sU9A)

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 2 - Loading and Viewing the Log Dataset
# MAGIC 
# MAGIC Given that our data is stored in the following mentioned path, let's load it into a DataFrame. We'll do this in steps. First, we'll use `sqlContext.read.text()` or `spark.read.text()` to read the text file. This will produce a DataFrame with a single string column called `value`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Taking a look at the metadata of our dataframe

# COMMAND ----------

# make sure you have upload NASA_access_log_Aug95.txt and NASA_access_log_Jul95.txt onto Spark before you run the following code

base_df = spark.read.text('dbfs:/FileStore/tables/*.txt')
base_df.printSchema()

# COMMAND ----------

type(base_df)

# COMMAND ----------

# MAGIC %md
# MAGIC You can also convert a dataframe to an RDD if needed

# COMMAND ----------

base_df_rdd = base_df.rdd
type(base_df_rdd)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Viewing sample data in our dataframe
# MAGIC Looks like it needs to be wrangled and parsed!

# COMMAND ----------

base_df.show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Getting data from an RDD is slightly different. You can see how the data representation is different in the following RDD

# COMMAND ----------

base_df_rdd.take(10)

# COMMAND ----------

# MAGIC %md
# MAGIC # Data transformation

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 1 - Data Wrangling
# MAGIC 
# MAGIC In this section, we will try and clean and parse our log dataset to really extract structured attributes with meaningful information from each log message.
# MAGIC 
# MAGIC ### Data understanding
# MAGIC If you're familiar with web server logs, you'll recognize that the above displayed data is in [Common Log Format](https://www.w3.org/Daemon/User/Config/Logging.html#common-logfile-format). 
# MAGIC 
# MAGIC The fields are:
# MAGIC __`remotehost rfc931 authuser [date] "request" status bytes`__
# MAGIC 
# MAGIC 
# MAGIC | field         | meaning                                                                |
# MAGIC | ------------- | ---------------------------------------------------------------------- |
# MAGIC | _remotehost_  | Remote hostname (or IP number if DNS hostname is not available or if [DNSLookup](https://www.w3.org/Daemon/User/Config/General.html#DNSLookup) is off).       |
# MAGIC | _rfc931_      | The remote logname of the user if at all it is present. |
# MAGIC | _authuser_    | The username of the remote user after authentication by the HTTP server.  |
# MAGIC | _[date]_      | Date and time of the request.                                      |
# MAGIC | _"request"_   | The request, exactly as it came from the browser or client.            |
# MAGIC | _status_      | The [HTTP status code](https://en.wikipedia.org/wiki/List_of_HTTP_status_codes) the server sent back to the client.               |
# MAGIC | _bytes_       | The number of bytes (`Content-Length`) transferred to the client.      |
# MAGIC 
# MAGIC We will need to use some specific techniques to parse, match and extract these attributes from the log data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Parsing and Extraction with Regular Expressions
# MAGIC 
# MAGIC Next, we have to parse it into individual columns. We'll use the special built-in [regexp\_extract()](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.regexp_extract)
# MAGIC function to do the parsing. This function matches a column against a regular expression with one or more [capture groups](http://regexone.com/lesson/capturing_groups) and allows you to extract one of the matched groups. We'll use one regular expression for each field we wish to extract.
# MAGIC 
# MAGIC You must have heard or used a fair bit of regular expressions by now. If you find regular expressions confusing (and they certainly _can_ be), and you want to learn more about them, we recommend checking out the
# MAGIC [RegexOne web site](http://regexone.com/). You might also find [_Regular Expressions Cookbook_](http://shop.oreilly.com/product/0636920023630.do), by Goyvaerts and Levithan, to be useful as a reference.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Let's take a look at our dataset dimensions

# COMMAND ----------

print((base_df.count(), len(base_df.columns)))

# COMMAND ----------

# MAGIC %md
# MAGIC Let's extract and take a look at some sample log messages

# COMMAND ----------

sample_logs = [item['value'] for item in base_df.take(15)]
sample_logs

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extracting host names
# MAGIC 
# MAGIC Let's try and write some regular expressions to extract the host name from the logs

# COMMAND ----------

host_pattern = r'(^\S+\.[\S+\.]+\S+)\s'
hosts = [re.search(host_pattern, item).group(1)
           if re.search(host_pattern, item)
           else 'no match'
           for item in sample_logs]
hosts

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extracting timestamps 
# MAGIC 
# MAGIC Let's now try and use regular expressions to extract the timestamp fields from the logs

# COMMAND ----------

ts_pattern = r'\[(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2} -\d{4})]'
timestamps = [re.search(ts_pattern, item).group(1) for item in sample_logs]
timestamps

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extracting HTTP Request Method, URIs and Protocol 
# MAGIC 
# MAGIC Let's now try and use regular expressions to extract the HTTP request methods, URIs and Protocol patterns fields from the logs

# COMMAND ----------

method_uri_protocol_pattern = r'\"(\S+)\s(\S+)\s*(\S*)\"'
method_uri_protocol = [re.search(method_uri_protocol_pattern, item).groups()
               if re.search(method_uri_protocol_pattern, item)
               else 'no match'
              for item in sample_logs]
method_uri_protocol

# COMMAND ----------

# MAGIC %md
# MAGIC ### Building an intermediate parsed dataframe
# MAGIC 
# MAGIC Let's try and use our regular expressions we have implemented so far into parsing and extracting the relevant entities in separate columns in a new dataframe

# COMMAND ----------

from pyspark.sql.functions import regexp_extract

logs_df = base_df.select(regexp_extract('value', host_pattern, 1).alias('host'),
                         regexp_extract('value', ts_pattern, 1).alias('timestamp'),
                         regexp_extract('value', method_uri_protocol_pattern, 1).alias('method'),
                         regexp_extract('value', method_uri_protocol_pattern, 2).alias('endpoint'),
                         regexp_extract('value', method_uri_protocol_pattern, 3).alias('protocol'))
logs_df.show(10, truncate=False)
print((logs_df.count(), len(logs_df.columns)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extracting HTTP Status Codes
# MAGIC 
# MAGIC Let's now try and use regular expressions to extract the HTTP status codes from the logs

# COMMAND ----------

status_pattern = r'\s(\d{3})\s'
status = [re.search(status_pattern, item).group(1) for item in sample_logs]
print(status)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extracting HTTP Response Content Size
# MAGIC 
# MAGIC Let's now try and use regular expressions to extract the HTTP response content size from the logs

# COMMAND ----------

content_size_pattern = r'\s(\d+)$'
content_size = [re.search(content_size_pattern, item).group(1) for item in sample_logs]
print(content_size)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Q1: Your Turn: Putting it all together 
# MAGIC 
# MAGIC Let's now try and leverage all the regular expression patterns we previously built and use the `regexp_extract(...)` method to build our dataframe with all the log attributes neatly extracted in their own separate columns.
# MAGIC 
# MAGIC - You can reuse the code we used previously to build the intermediate dataframe
# MAGIC - Remember to cast the HTTP status code and content size as integers. 
# MAGIC - You can cast data as integer type using the following: __`regexp_extract('value', ...., ...).cast('integer').alias(...)`__

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code

logs_df = base_df.select(regexp_extract('value', host_pattern, 1).alias('host'),
                         regexp_extract('value', ts_pattern, 1).alias('timestamp'),
                         regexp_extract('value', method_uri_protocol_pattern, 1).alias('method'),
                         regexp_extract('value', method_uri_protocol_pattern, 2).alias('endpoint'),
                         regexp_extract('value', method_uri_protocol_pattern, 3).alias('protocol'),
                         regexp_extract('value', status_pattern, 1).cast('integer').alias('status'),
                         regexp_extract('value', content_size_pattern, 1).alias('content_size')
                        )
logs_df.show(10, truncate=True)
print((logs_df.count(), len(logs_df.columns)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Finding Missing Values
# MAGIC 
# MAGIC Missing and null values are the bane of data analysis and machine learning. Let's see how well our data parsing and extraction logic worked. First, let's verify that there are no null rows in the original dataframe.

# COMMAND ----------

base_df.filter(base_df['value'].isNull()).count()

# COMMAND ----------

# MAGIC %md
# MAGIC If our data parsing and extraction worked properly, we should not have any rows with potential null values. Let's try and put that to test!

# COMMAND ----------

bad_rows_df = logs_df.filter(logs_df['host'].isNull()| 
                             logs_df['timestamp'].isNull() | 
                             logs_df['method'].isNull() |
                             logs_df['endpoint'].isNull() |
                             logs_df['status'].isNull() |
                             logs_df['content_size'].isNull()|
                             logs_df['protocol'].isNull())
bad_rows_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC Ouch! Looks like we have over 30K missing values in our data! Can we handle this?

# COMMAND ----------

# MAGIC %md
# MAGIC Do remember, this is not a regular pandas dataframe which you can directly query and get which columns have null. Our so-called _big dataset_ is residing on disk which can potentially be present in multiple nodes in a spark cluster. So how do we find out which columns have potential nulls? 
# MAGIC 
# MAGIC ### Finding Null Counts
# MAGIC 
# MAGIC We can typically use the following technique to find out which columns have null values. 
# MAGIC 
# MAGIC (__Note:__ This approach is adapted from an [excellent answer](http://stackoverflow.com/a/33901312) on StackOverflow.)

# COMMAND ----------

logs_df.columns

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.functions import sum as spark_sum

def count_null(col_name):
    return spark_sum(col(col_name).isNull().cast('integer')).alias(col_name)

# Build up a list of column expressions, one per column.
exprs = [count_null(col_name) for col_name in logs_df.columns]

# Run the aggregation. The *exprs converts the list of expressions into
# variable function arguments.
logs_df.agg(*exprs).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Well, looks like we have one missing value in the `status` column and everything else is in the `content_size` column. 
# MAGIC Let's see if we can figure out what's wrong!

# COMMAND ----------

# MAGIC %md
# MAGIC ### Handling nulls in HTTP status
# MAGIC 
# MAGIC If you had solved it correctly, our original parsing regular expression for the `status` column was:
# MAGIC 
# MAGIC ```
# MAGIC regexp_extract('value', r'\s(\d{3})\s', 1).cast('integer').alias('status')
# MAGIC ``` 
# MAGIC 
# MAGIC Could it be that there are more digits making our regular expression wrong? or is the data point itself bad? Let's try and find out!
# MAGIC 
# MAGIC **Note**: In the expression below, `~` means "not".

# COMMAND ----------

null_status_df = base_df.filter(~base_df['value'].rlike(r'\s(\d{3})\s'))
null_status_df.count()

# COMMAND ----------

null_status_df.show(truncate=False)

# COMMAND ----------

bad_status_df = null_status_df.select(regexp_extract('value', host_pattern, 1).alias('host'),
                                      regexp_extract('value', ts_pattern, 1).alias('timestamp'),
                                      regexp_extract('value', method_uri_protocol_pattern, 1).alias('method'),
                                      regexp_extract('value', method_uri_protocol_pattern, 2).alias('endpoint'),
                                      regexp_extract('value', method_uri_protocol_pattern, 3).alias('protocol'),
                                      regexp_extract('value', status_pattern, 1).cast('integer').alias('status'),
                                      regexp_extract('value', content_size_pattern, 1).cast('integer').alias('content_size'))
bad_status_df.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Looks like the record itself is an incomplete record with no useful information, the best option would be to drop this record as follows!

# COMMAND ----------

logs_df.count()

# COMMAND ----------

logs_df = logs_df[logs_df['status'].isNotNull()]
logs_df.count()

# COMMAND ----------

exprs = [count_null(col_name) for col_name in logs_df.columns]
logs_df.agg(*exprs).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Handling nulls in HTTP content size
# MAGIC 
# MAGIC Again based on our previous regular expression and assuming you were able to solve it correctly, our original parsing regular expression for the `content_size` column was:
# MAGIC 
# MAGIC ```
# MAGIC regexp_extract('value', r'\s(\d+)$', 1).cast('integer').alias('content_size')
# MAGIC ``` 
# MAGIC 
# MAGIC Could there be missing data in our original dataset itself? Let's try and find out!

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q2: Your Turn: Find out the records in our base data frame with potential missing content sizes
# MAGIC 
# MAGIC - Use the `r'\s\d+$'` regex pattern with the `rlike()` function like we demonstrated in the previous example
# MAGIC - Remember to work on `base_df` since we are searching on the raw records NOT the parsed `logs_df`
# MAGIC - Find the total count of the records with missing content size in `base_df` using the `count()` function

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code

null_content_size_df = base_df.filter(~base_df['value'].rlike(r'\s\d+$'))
null_content_size_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q3: Your Turn: Display the top ten records of your data frame having missing content sizes

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code

null_content_size_df.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Assuming you were able to get to the missing records above, it is quite evident that the bad raw data records correspond to error responses, where no content was sent back and the server emitted a "`-`" for the `content_size` field. 
# MAGIC 
# MAGIC Since we don't want to discard those rows from our analysis, let's impute or fill them to 0.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Q4: Your Turn: Fix the rows with null content\_size
# MAGIC 
# MAGIC The easiest solution is to replace the null values in `logs_df` with 0 like we discussed earlier. The Spark DataFrame API provides a set of functions and fields specifically designed for working with null values, among them:
# MAGIC 
# MAGIC * [fillna()](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.fillna), which fills null values with specified non-null values.
# MAGIC * [na](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.na), which returns a [DataFrameNaFunctions](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrameNaFunctions) object with many functions for operating on null columns.
# MAGIC 
# MAGIC There are several ways to invoke this function. The easiest is just to replace _all_ null columns with known values. But, for safety, it's better to pass a Python dictionary containing (column\_name, value) mappings. That's what we'll do. A sample example from the documentation is depicted below
# MAGIC 
# MAGIC ```
# MAGIC >>> df4.na.fill({'age': 50, 'name': 'unknown'}).show()
# MAGIC +---+------+-------+
# MAGIC |age|height|   name|
# MAGIC +---+------+-------+
# MAGIC | 10|    80|  Alice|
# MAGIC |  5|  null|    Bob|
# MAGIC | 50|  null|    Tom|
# MAGIC | 50|  null|unknown|
# MAGIC +---+------+-------+
# MAGIC ```
# MAGIC 
# MAGIC Now use this function and fill all the missing values in the `content_size` field with 0!

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code

logs_df = logs_df.na.fill({'content_size':0})

# COMMAND ----------

# MAGIC %md
# MAGIC Now assuming you were able to fill in the missing values successfully in the previous question, we should have no missing values \ nulls in our dataset. Let's verify this!

# COMMAND ----------

exprs = [count_null(col_name) for col_name in logs_df.columns]
logs_df.agg(*exprs).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Look at that, no missing values!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Handling Temporal Fields (Timestamp)
# MAGIC 
# MAGIC Now that we have a clean, parsed DataFrame, we have to parse the timestamp field into an actual timestamp. The Common Log Format time is somewhat non-standard. A User-Defined Function (UDF) is the most straightforward way to parse it.

# COMMAND ----------

from pyspark.sql.functions import udf

month_map = {
  'Jan': 1, 'Feb': 2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7,
  'Aug':8,  'Sep': 9, 'Oct':10, 'Nov': 11, 'Dec': 12
}

def parse_clf_time(text):
    """ Convert Common Log time format into a Python datetime object
    Args:
        text (str): date and time in Apache time format [dd/mmm/yyyy:hh:mm:ss (+/-)zzzz]
    Returns:
        a string suitable for passing to CAST('timestamp')
    """
    # NOTE: We're ignoring time zone here. In a production application, you'd want to handle that.
    return "{0:04d}-{1:02d}-{2:02d} {3:02d}:{4:02d}:{5:02d}".format(
      int(text[7:11]),
      month_map[text[3:6]],
      int(text[0:2]),
      int(text[12:14]),
      int(text[15:17]),
      int(text[18:20])
    )

# COMMAND ----------

sample_ts = [item['timestamp'] for item in logs_df.select('timestamp').take(5)]
sample_ts

# COMMAND ----------

[parse_clf_time(item) for item in sample_ts]

# COMMAND ----------

udf_parse_time = udf(parse_clf_time)

logs_df = logs_df.select('*', udf_parse_time(logs_df['timestamp']).cast('timestamp').alias('time')).drop('timestamp')
logs_df.show(10, truncate=True)

# COMMAND ----------

logs_df.printSchema()

# COMMAND ----------

logs_df.limit(5).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's now cache `logs_df` since we will be using it extensively for our data analysis section in the next part!

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 2 - Exploratory Data Analysis on our Web Logs
# MAGIC 
# MAGIC Now that we have a DataFrame containing the parsed log file as a data frame, we can perform some interesting exploratory data analysis (EDA)
# MAGIC 
# MAGIC ## Example: Content Size Statistics
# MAGIC 
# MAGIC Let's compute some statistics about the sizes of content being returned by the web server. In particular, we'd like to know what are the average, minimum, and maximum content sizes.
# MAGIC 
# MAGIC We can compute the statistics by calling `.describe()` on the `content_size` column of `logs_df`.  The `.describe()` function returns the count, mean, stddev, min, and max of a given column.

# COMMAND ----------

content_size_summary_df = logs_df.describe(['content_size'])
content_size_summary_df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC Alternatively, we can use SQL to directly calculate these statistics.  You can explore many useful functions within the `pyspark.sql.functions` module in the [documentation](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#module-pyspark.sql.functions).
# MAGIC 
# MAGIC After we apply the `.agg()` function, we call `toPandas()` to extract and convert the result into a `pandas` dataframe which has better formatting on Jupyter notebooks

# COMMAND ----------

from pyspark.sql import functions as F

(logs_df.agg(F.min(logs_df['content_size']).alias('min_content_size'),
             F.max(logs_df['content_size']).alias('max_content_size'),
             F.mean(logs_df['content_size']).alias('mean_content_size'),
             F.stddev(logs_df['content_size']).alias('std_content_size'),
             F.count(logs_df['content_size']).alias('count_content_size'))
        .toPandas())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example: HTTP Status Code Analysis
# MAGIC 
# MAGIC Next, let's look at the status code values that appear in the log. We want to know which status code values appear in the data and how many times.  
# MAGIC 
# MAGIC We again start with `logs_df`, then group by the `status` column, apply the `.count()` aggregation function, and sort by the `status` column.

# COMMAND ----------

status_freq_df = (logs_df
                     .groupBy('status')
                     .count()
                     .sort('status')
                     .cache())

# COMMAND ----------

print('Total distinct HTTP Status Codes:', status_freq_df.count())

# COMMAND ----------

status_freq_pd_df = status_freq_df.toPandas()
status_freq_pd_df

# COMMAND ----------

import matplotlib.pyplot as plt
%matplotlib inline

status_freq_pd_df.plot(x='status', y='count', kind='bar')

# COMMAND ----------

log_freq_df = status_freq_df.withColumn('log(count)', F.log(status_freq_df['count']))
log_freq_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q5: Your Turn: Convert the log\_freq\_df to a pandas DataFrame and plot a bar chart displaying counts of each HTTP Status Code

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code
log_freq_pd_df = log_freq_df.toPandas()
log_freq_pd_df.plot(x='status', y='count', kind='bar')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Your Turn: Q6: Analyzing Frequent Hosts
# MAGIC 
# MAGIC Let's look at hosts that have accessed the server frequently. Try to get the count of total accesses by each `host` and then sort by the counts and display only the top ten most frequent hosts.
# MAGIC 
# MAGIC __Hints:__
# MAGIC 
# MAGIC - Your Spark DataFrame has a `host` column
# MAGIC - Get the counts per `host` which would make a `count` column
# MAGIC - Sort by the counts. Please check [__the documentation__](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.sort) to see how to sort in reverse
# MAGIC - Remember only to get the top 10 rows from the aggregated dataframe and show them

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code

host_sum_df =(logs_df
                     .groupBy('host')
                     .count()
                     .sort('count', ascending = False)
                     .limit(10)
                     .cache())

host_sum_df.show(truncate=False)

# COMMAND ----------

host_sum_pd_df = host_sum_df.toPandas()
host_sum_pd_df.iloc[8]['host']

# COMMAND ----------

# MAGIC %md
# MAGIC Looks like we have some empty strings as one of the top host names! This teaches us a valuable lesson to not just check for nulls but also potentially empty strings when data wrangling.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Your Turn: Q7: Display the Top 20 Frequent EndPoints
# MAGIC 
# MAGIC Now, let's visualize the number of hits to endpoints (URIs) in the log. To perform this task, start with our `logs_df` and group by the `endpoint` column, aggregate by count, and sort in descending order like the previous question. Also remember to show only the top 20 most frequently accessed endpoints

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code

paths_df = (logs_df
                     .groupBy('endpoint')
                     .count()
                     .sort('count', ascending = False)
                     .limit(20)
                     .cache())

paths_df.show(truncate=False)

# COMMAND ----------

paths_pd_df = paths_df.toPandas()
paths_pd_df

# COMMAND ----------

paths_pd_df.plot(x='endpoint', y='count', kind='bar')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Your Turn: Q8: Top Ten Error Endpoints
# MAGIC 
# MAGIC What are the top ten endpoints requested which did not have return code 200 (HTTP Status OK)? 
# MAGIC 
# MAGIC Create a sorted list containing the endpoints and the number of times that they were accessed with a non-200 return code and show the top ten.
# MAGIC 
# MAGIC Think about the steps that you need to perform to determine which endpoints did not have a 200 return code (combination of filtering, grouping, sorting and selecting the top ten aggregated records)

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code

not200_df = (logs_df
                     .filter(logs_df.status != 200)
                     .cache())

error_endpoints_freq_df = (not200_df
                               .groupBy('endpoint')
                               .count()
                               .sort('count', ascending = False)
                               .limit(10)
                               .cache()
                          )

# COMMAND ----------

error_endpoints_freq_df.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example: Number of Unique Daily Hosts
# MAGIC 
# MAGIC For an advanced example, let's look at a way to determine the number of unique hosts in the entire log on a day-by-day basis. This computation will give us counts of the number of unique daily hosts. 
# MAGIC 
# MAGIC We'd like a DataFrame sorted by increasing day of the month which includes the day of the month and the associated number of unique hosts for that day. 
# MAGIC 
# MAGIC Think about the steps that you need to perform to count the number of different hosts that make requests *each* day.
# MAGIC *Since the log only covers a single month, you can ignore the month.*  You may want to use the [`dayofmonth` function](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.dayofmonth) in the `pyspark.sql.functions` module (which we have already imported as __`F`__.
# MAGIC 
# MAGIC 
# MAGIC **`host_day_df`**
# MAGIC 
# MAGIC A DataFrame with two columns
# MAGIC 
# MAGIC | column | explanation          |
# MAGIC | ------ | -------------------- |
# MAGIC | `host` | the host name        |
# MAGIC | `day`  | the day of the month |
# MAGIC 
# MAGIC There will be one row in this DataFrame for each row in `logs_df`. Essentially, we are just transforming each row of `logs_df`. For example, for this row in `logs_df`:
# MAGIC 
# MAGIC ```
# MAGIC unicomp6.unicomp.net - - [01/Aug/1995:00:35:41 -0400] "GET /shuttle/missions/sts-73/news HTTP/1.0" 302 -
# MAGIC ```
# MAGIC 
# MAGIC your `host_day_df` should have:
# MAGIC 
# MAGIC ```
# MAGIC unicomp6.unicomp.net 1
# MAGIC ```

# COMMAND ----------

host_day_df = logs_df.select(logs_df.host, 
                             F.dayofmonth('time').alias('day'))
host_day_df.show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC **`host_day_distinct_df`**
# MAGIC 
# MAGIC This DataFrame has the same columns as `host_day_distinct_df`, but with duplicate (`day`, `host`) rows removed.

# COMMAND ----------

host_day_distinct_df = (host_day_df
                          .dropDuplicates())
host_day_distinct_df.show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC **`daily_unique_hosts_df`**
# MAGIC 
# MAGIC A DataFrame with two columns:
# MAGIC 
# MAGIC | column  | explanation                                        |
# MAGIC | ------- | -------------------------------------------------- |
# MAGIC | `day`   | the day of the month                               |
# MAGIC | `count` | the number of unique requesting hosts for that day |

# COMMAND ----------

daily_hosts_df = (host_day_distinct_df
                     .groupBy('day')
                     .count()
                     .sort("day"))
daily_hosts_df = daily_hosts_df.toPandas()
daily_hosts_df.T

# COMMAND ----------

daily_hosts_df.plot(x='day', y='count', kind='line')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Your Turn: Q9: Counting 404 Response Codes
# MAGIC 
# MAGIC Create a DataFrame containing only log records with a 404 status code (Not Found). 
# MAGIC 
# MAGIC Make sure you `cache()` the `not_found_df` dataframe as we will use it in the rest of the exercises here.
# MAGIC 
# MAGIC __How many 404 records are in the log?__

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code

not_found_df = (logs_df
                     .filter(logs_df.status == 404)
                     .cache())
print(('Total 404 responses: {}').format(not_found_df))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Your Turn: Q10: Listing the Top Twenty 404 Response Code Endpoints
# MAGIC 
# MAGIC Using the DataFrame containing only log records with a 404 response code that you cached in Q9, print out a list of the top twenty endpoints that generate the most 404 errors.
# MAGIC 
# MAGIC *Remember, top endpoints should be in sorted order*

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code

hosts_404_count_df = (not_found_df
                          .groupBy('endpoint')
                               .count()
                               .sort('count', ascending = False)
                               .limit(20)
                               .cache())

hosts_404_count_df.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Your Turn: Q11: Visualizing 404 Errors per Day
# MAGIC 
# MAGIC Let's explore the 404 records temporally now. Similar to the example showing the number of unique daily hosts, break down the 404 requests by day and get the daily counts sorted by day in `errors_by_date_sorted_df`.
# MAGIC 
# MAGIC - Display the results as a pandas dataframe 
# MAGIC - Also visualize the same dataframe then as a line chart

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code

errors_by_date_sorted_df = (not_found_df
                               .groupBy(F.dayofmonth('time').alias('day'))
                               .count()
                               .sort('day', ascending = True)
                               .cache())

errors_by_date_sorted_df = errors_by_date_sorted_df.toPandas()
errors_by_date_sorted_df.T

# COMMAND ----------

errors_by_date_sorted_df.plot(x='day', y='count', kind='line')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Your Turn: Q12: Visualizing Hourly 404 Errors
# MAGIC 
# MAGIC Using the DataFrame `not_found_df` you cached in the Q10, group and sort by hour of the day in increasing order, to create a DataFrame containing the total number of 404 responses for HTTP requests for each hour of the day (midnight starts at 0). 
# MAGIC 
# MAGIC - Remember to check out the [__hour__](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.functions.hour) function and use it (we have already imported __`pyspark.sql.functions`__ as __`F`__ earlier
# MAGIC - Output should be a bar graph displaying the total number of 404 errors per hour

# COMMAND ----------

# TODO: Replace <FILL IN> with appropriate code

hourly_avg_errors_sorted_df = (not_found_df
                               .groupBy(F.hour('time').alias('hour'))
                               .count()
                               .sort('hour', ascending = True)
                               .cache())

# COMMAND ----------

hourly_avg_errors_sorted_df.toPandas().plot(x='hour', y='count', kind='bar')

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Data

# COMMAND ----------

# MAGIC %md
# MAGIC Up till now, you have completed data extraction, data transformation, and some exploratory data analysis. In the end of this project, we will complete the last step of ETL process: data loading, so the data after  your processing, wrangling, cleaning, can be used by either yourself or other colleagues later. Since we have gone through a few iteration of data processing and data wrangling, it is a good idea to make sure which one is the current dataframe you want to store and load.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Your Turn: Q13: Check data integrity before loading

# COMMAND ----------

# TODO: Review the data frame you will like to store and load. Replace <FILL IN> with appropriate code

print(logs_df.count())
print(logs_df.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC To save your dataframe in CSV file format, you call simply replace the name of the dataframe and assign file name in the following:

# COMMAND ----------

# MAGIC %md
# MAGIC ## Your Turn: Q14: Save your data as a CSV file

# COMMAND ----------

# TODO: Review the data frame you will like to store and load. Replace <FILL IN> with appropriate code

logs_df.write.save("NASA_final_logs_v2_df.csv", format = 'csv')

# COMMAND ----------

# TODO: Check to see if you have stored and loaded the CSV file successfully by checking the first 5 rows. Replace <FILL IN> with appropriate code
#spark_session\

spark\
	.sparkContext\
	.textFile("NASA_final_logs_v2_df.csv")\
	.take(5)

# COMMAND ----------

# MAGIC %md
# MAGIC Similarly, you can also store and load your dataframe as a JSON file by completing the following:

# COMMAND ----------

# MAGIC %md
# MAGIC ## Your Turn: Q15: Save your data as a JSON file

# COMMAND ----------

# TODO: Review the data frame you will like to store and load. Replace <FILL IN> with appropriate code

logs_df.write.save("NASA_final_logs_v2_df.json", format = 'json')

# COMMAND ----------

# TODO: Similarly, check the first 5 rows in the JSON file. Replace <FILL IN> with appropriate code

spark\
	.sparkContext\
	.textFile("NASA_final_logs_v2_df.json")\
	.take(5)

# COMMAND ----------

# MAGIC %md
# MAGIC There is a lot more you can do about data storing and loading in terms of data formats and settings. Check out more about these options [__here__](https://spark.apache.org/docs/latest/sql-data-sources-load-save-functions.html).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Congratulations! You have finished the mini-project for this unit!
