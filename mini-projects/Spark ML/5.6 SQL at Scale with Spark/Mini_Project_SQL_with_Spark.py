# Databricks notebook source
# MAGIC %md
# MAGIC ## SQL at Scale with Spark SQL
# MAGIC 
# MAGIC Welcome to the SQL mini project. For this project, you will use the Databricks Platform and work through a series of exercises using Spark SQL. The dataset size may not be too big but the intent here is to familiarize yourself with the Spark SQL interface which scales easily to huge datasets, without you having to worry about changing your SQL queries. 
# MAGIC 
# MAGIC The data you need is present in the mini-project folder in the form of three CSV files. This data will be imported in Databricks to create the following tables under the __`country_club`__ database.
# MAGIC 
# MAGIC <br>
# MAGIC 1. The __`bookings`__ table,
# MAGIC 2. The __`facilities`__ table, and
# MAGIC 3. The __`members`__ table.
# MAGIC 
# MAGIC You will be uploading these datasets shortly into the Databricks platform to understand how to create a database within minutes! Once the database and the tables are populated, you will be focusing on the mini-project questions.
# MAGIC 
# MAGIC In the mini project, you'll be asked a series of questions. You can solve them using the databricks platform, but for the final deliverable,
# MAGIC please download this notebook as an IPython notebook (__`File -> Export -> IPython Notebook`__) and upload it to your GitHub.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating the Database
# MAGIC 
# MAGIC We will first create our database in which we will be creating our three tables of interest

# COMMAND ----------

# MAGIC %sql 
# MAGIC drop database if exists country_club cascade;
# MAGIC create database country_club;
# MAGIC show databases;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Creating the Tables
# MAGIC 
# MAGIC In this section, we will be creating the three tables of interest and populate them with the data from the CSV files already available to you. 
# MAGIC To get started, first upload the three CSV files to the DBFS as depicted in the following figure
# MAGIC 
# MAGIC ![](https://i.imgur.com/QcCruBr.png)
# MAGIC 
# MAGIC 
# MAGIC Once you have done this, please remember to execute the following code to build the dataframes which will be saved as tables in our database

# COMMAND ----------

# File location and type
file_location_bookings = "/FileStore/tables/Bookings.csv"
file_location_facilities = "/FileStore/tables/Facilities.csv"
file_location_members = "/FileStore/tables/Members.csv"

file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
bookings_df = (spark.read.format(file_type) 
                    .option("inferSchema", infer_schema) 
                    .option("header", first_row_is_header) 
                    .option("sep", delimiter) 
                    .load(file_location_bookings))

facilities_df = (spark.read.format(file_type) 
                      .option("inferSchema", infer_schema) 
                      .option("header", first_row_is_header) 
                      .option("sep", delimiter) 
                      .load(file_location_facilities))

members_df = (spark.read.format(file_type) 
                      .option("inferSchema", infer_schema) 
                      .option("header", first_row_is_header) 
                      .option("sep", delimiter) 
                      .load(file_location_members))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Viewing the dataframe schemas
# MAGIC 
# MAGIC We can take a look at the schemas of our potential tables to be written to our database soon

# COMMAND ----------

print('Bookings Schema')
bookings_df.printSchema()
print('Facilities Schema')
facilities_df.printSchema()
print('Members Schema')
members_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create permanent tables
# MAGIC We will be creating three permanent tables here in our __`country_club`__ database as we discussed previously with the following code

# COMMAND ----------

permanent_table_name_bookings = "country_club.Bookings"
bookings_df.write.format("parquet").saveAsTable(permanent_table_name_bookings)

permanent_table_name_facilities = "country_club.Facilities"
facilities_df.write.format("parquet").saveAsTable(permanent_table_name_facilities)

permanent_table_name_members = "country_club.Members"
members_df.write.format("parquet").saveAsTable(permanent_table_name_members)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Refresh tables and check them

# COMMAND ----------

# MAGIC %sql
# MAGIC use country_club;
# MAGIC REFRESH table bookings;
# MAGIC REFRESH table facilities;
# MAGIC REFRESH table members;
# MAGIC show tables;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test a sample SQL query
# MAGIC 
# MAGIC __Note:__ You can use __`%sql`__ at the beginning of a cell and write SQL queries directly as seen in the following cell. Neat isn't it!

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from bookings order by starttime, bookid, facid, memid, slots limit 100 

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from members limit 3

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from facilities limit 3

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q1: Some of the facilities charge a fee to members, but some do not. Please list the names of the facilities that do.

# COMMAND ----------

# MAGIC %sql
# MAGIC select name --, membercost 
# MAGIC from facilities where membercost = 0

# COMMAND ----------

# MAGIC %md
# MAGIC ####  Q2: How many facilities do not charge a fee to members?

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(name) from facilities where membercost = 0

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q3: How can you produce a list of facilities that charge a fee to members, where the fee is less than 20% of the facility's monthly maintenance cost? 
# MAGIC #### Return the facid, facility name, member cost, and monthly maintenance of the facilities in question.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- This question was really not clear. Did it refer to 
# MAGIC -- facilities for whom the fee was less than 20% of the facility' monthly maintenance cost?
# MAGIC -- Or all facilities where the total member cost collected was less than 
# MAGIC -- 20% of the facilities monthly maintenance cost?
# MAGIC 
# MAGIC select facid, name, membercost, monthlymaintenance
# MAGIC from facilities
# MAGIC where membercost < monthlymaintenance * 0.2

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q4: How can you retrieve the details of facilities with ID 1 and 5? Write the query without using the OR operator.

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select *
# MAGIC from facilities where facid in (1,5)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q5: How can you produce a list of facilities, with each labelled as 'cheap' or 'expensive', depending on if their monthly maintenance cost is more than $100? 
# MAGIC #### Return the name and monthly maintenance of the facilities in question.

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select name, 
# MAGIC   (case when monthlymaintenance > 100 then 'expensive' else 'cheap' end) as monthlymaintenancelabel,
# MAGIC   monthlymaintenance
# MAGIC from facilities

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q6: You'd like to get the first and last name of the last member(s) who signed up. Do not use the LIMIT clause for your solution.

# COMMAND ----------

# MAGIC %sql
# MAGIC select firstname, surname --, joindate 
# MAGIC from members
# MAGIC where joindate in (select max(joindate) as latest_join_date from members)

# COMMAND ----------

# MAGIC %md
# MAGIC ####  Q7: How can you produce a list of all members who have used a tennis court?
# MAGIC - Include in your output the name of the court, and the name of the member formatted as a single column. 
# MAGIC - Ensure no duplicate data
# MAGIC - Also order by the member name.

# COMMAND ----------

# MAGIC %sql
# MAGIC select distinct f.name, concat(m.firstname," ",m.surname) as member_fullname
# MAGIC from facilities f 
# MAGIC join bookings b on (f.facid = b.facid)
# MAGIC join members m on (b.memid = m.memid)
# MAGIC where f.name like ('%Tennis Court%')
# MAGIC order by 2

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q8: How can you produce a list of bookings on the day of 2012-09-14 which will cost the member (or guest) more than $30? 
# MAGIC 
# MAGIC - Remember that guests have different costs to members (the listed costs are per half-hour 'slot')
# MAGIC - The guest user's ID is always 0. 
# MAGIC 
# MAGIC #### Include in your output the name of the facility, the name of the member formatted as a single column, and the cost.
# MAGIC 
# MAGIC - Order by descending cost, and do not use any subqueries.

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC 
# MAGIC select f.name, 
# MAGIC   concat(m.firstname," ",m.surname) as member_fullname, 
# MAGIC    sum(case when b.memid = 0 then f.guestcost * b.slots else f.membercost * b.slots end) as total_cost
# MAGIC from bookings b 
# MAGIC join facilities f on (b.facid = f.facid)  
# MAGIC join members m on (b.memid = m.memid )
# MAGIC where date_trunc('day',b.starttime) = '2012-09-14'
# MAGIC group by 1, 2
# MAGIC having total_cost>30
# MAGIC order by 3 desc

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q9: This time, produce the same result as in Q8, but using a subquery.

# COMMAND ----------

# MAGIC %sql
# MAGIC select f.name, 
# MAGIC   concat(m.firstname," ",m.surname) as member_fullname,
# MAGIC   t1.total_sum
# MAGIC from
# MAGIC     (select b.facid, b.memid, 
# MAGIC         sum(case when b.memid = 0 then f.guestcost * b.slots else f.membercost * b.slots end) as total_sum
# MAGIC       from bookings b 
# MAGIC       join facilities f on (b.facid = f.facid) 
# MAGIC       where date_trunc('day',b.starttime) = '2012-09-14'
# MAGIC       group by 1,2
# MAGIC       having total_sum > 30
# MAGIC       order by 3 desc
# MAGIC       ) t1
# MAGIC left join members m on (t1.memid = m.memid)
# MAGIC left join facilities f on (t1.facid = f.facid)
# MAGIC order by 3 desc

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q10: Produce a list of facilities with a total revenue less than 1000.
# MAGIC - The output should have facility name and total revenue, sorted by revenue. 
# MAGIC - Remember that there's a different cost for guests and members!

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select f.name, 
# MAGIC    sum(case when b.memid = 0 then f.guestcost * b.slots else f.membercost * b.slots end) as total_revenue
# MAGIC from bookings b 
# MAGIC join facilities f on (b.facid = f.facid)  
# MAGIC join members m on (b.memid = m.memid)
# MAGIC group by 1
# MAGIC having total_revenue<1000
# MAGIC order by 2
