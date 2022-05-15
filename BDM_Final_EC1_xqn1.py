from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, IntegerType, MapType, StringType, FloatType, ArrayType

import sys

def main(sc):
  import csv
  import json
  import shapely
  import pyproj
  from pyproj import Transformer
  from shapely.geometry import Point

  spark = SparkSession(sc)

  # READ MAIN DATASET

  df = spark.read.csv('/tmp/bdm/weekly-patterns-nyc-2019-2020/*',
                    header=True,
                    escape='\"') \
                    .select('placekey','poi_cbg','visitor_home_cbgs','date_range_start','date_range_end') \
                    .cache()


 # STEP 1: FILTER MAIN DATAFRAME FOR VISITS TO NYC SUPERMARKETS USING nyc_supermarkets.csv

  # Read nyc_supermarkets.csv and take columns of interest: safegraph_placekey (to match to main dataset), latitude and longitude
  df_supermarkets = spark.read.csv('nyc_supermarkets.csv', header=True) \
        .select('safegraph_placekey',
        F.col('latitude').alias('supermarkets_x').cast('float'),
        F.col('longitude').alias('supermarkets_y').cast('float')) \
        .cache()

  # Inner join and filter main dataframe with df_supermarkets on placekeys
  df2 = df.join(df_supermarkets, df['placekey']==df_supermarkets['safegraph_placekey'], how='inner').cache()


  # STEP 2: FILTER FOR VISITS OVERLAPPING WITH 4 MONTHS OF INTEREST
  dates = {'2019-03','2019-10','2020-03','2020-10'}

  # Extract months from date_range_start and date_range_end
  df2 = df2.withColumn('start',df.date_range_start.substr(1,7)) \
         .withColumn('end',df.date_range_end.substr(1,7)) \
         .where(F.col('start').isin(dates) | F.col('end').isin(dates))

  # Create udf to retain only one date from both columns
  udf_dates = F.udf(lambda x,y: x if x in dates else y)
  df3 = df2.withColumn('date', udf_dates(F.col('start'),F.col('end'))) \
    .select('placekey','poi_cbg','visitor_home_cbgs','date','supermarkets_x','supermarkets_y') \
    .cache()



  # STEP 3: FILTER FOR VISITS IN nyc_cbg_centroids.csv

  # Explode json inside visitor_home_cbgs column to multiple rows
  # Using method from FAQ section of Safegraph Docs

  def parser(element):
    return json.loads(element)

  jsonudf = F.udf(parser, MapType(StringType(), IntegerType()))

  visitor_home_cbgs_parsed = df3.withColumn('parsed_visitor_home_cbgs', jsonudf('visitor_home_cbgs'))
  visitor_home_cbgs_exploded = visitor_home_cbgs_parsed.select('placekey','poi_cbg', 'date', 'supermarkets_x','supermarkets_y', F.explode('parsed_visitor_home_cbgs'))

 # Read nyc_cbg_centroids.csv and cast latitude/longitude to floats
  df_cbgs = spark.read.csv('nyc_cbg_centroids.csv', header=True) \
                      .select('cbg_fips',
        F.col('latitude').alias('cbg_x').cast('float'),
        F.col('longitude').alias('cbg_y').cast('float'))

  # Join main dataframe with df_cbgs on cbg_fips
  df4 = visitor_home_cbgs_exploded.join(df_cbgs, visitor_home_cbgs_exploded['key']==df_cbgs['cbg_fips'], how='inner').cache()

  df4 = df4.select('cbg_fips','date',
                  df4['value'].alias('visitor_count'),
                  'supermarkets_x','supermarkets_y','cbg_x','cbg_y') \
                  .cache()


  # STEP 4: REPROJECT COORDINATES AND CALCULATE DISTANCE
  # Define function for transforming to 2263 and calculating distance
  def obtain_distance(x, y, a, b):

    t = Transformer.from_crs(4326,2263)
    new_x, new_y = t.transform(x,y)
    new_a, new_b = t.transform(a,b)
    dist = Point(new_x,new_y).distance(Point((new_a,new_b)))/5280
    return round(dist, 2)

  # Create udf
  udf_dist = F.udf(obtain_distance)

 # Calculate distances
  df5 = df4.withColumn('dist', udf_dist(F.col('supermarkets_x'), F.col('supermarkets_y'), F.col('cbg_x'), F.col('cbg_y'))) \
  .select('cbg_fips','date','visitor_count',
          F.col('dist').cast(FloatType())).cache()



  # EXTRA CREDIT 1: CALCULATE MEDIAN

  months = ['2019-03','2019-10','2020-03','2020-10']

  # Create udf for making array of length = visitor_count for exploding
  # Obtain 1 row per visitor
  # Approximate median at 0.5 percentile

  n_to_array = F.udf(lambda x: [x] * x, ArrayType(IntegerType()))

  df5a = df5.withColumn('n', n_to_array(df5.visitor_count)) \
         .withColumn('n', F.explode(F.col('n'))) \
    .groupBy('cbg_fips','date') \
    .agg(F.expr('percentile(dist, array(0.5))')[0].alias('median_dist')) \
    .select('cbg_fips','date',F.round(F.col('median_dist').cast(FloatType()),2).alias('median_dist')) \
    .groupBy('cbg_fips').pivot('date',months).max('median_dist') \
    .sort('cbg_fips') \
    .write.option('header','true') \
    .csv(sys.argv[1] if len(sys.argv)>1 else 'output_ec1')

  
if __name__=="__main__":
  sc = SparkContext()
  main(sc)