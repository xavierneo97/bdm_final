from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, IntegerType, MapType, StringType, FloatType

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



  # STEP 1: FILTER FOR VISITS TO NYC SUPERMARKETS USING nyc_supermarkets.csv 'placekeys'

  # Read nyc_supermarkets.csv and take columns of interest: safegraph_placekey (to match to main dataset), latitude and longitude
  df_supermarkets = spark.read.csv('nyc_supermarkets.csv', header=True).cache()

  df_supermarkets = df_supermarkets.select('safegraph_placekey',
        df_supermarkets['latitude'].cast('float'),
        df_supermarkets['longitude'].cast('float')) \
        .cache()

  # Create set of NYC supermarket safegraph placekeys
  set_supermarkets = set(list(df_supermarkets.select('safegraph_placekey').toPandas()['safegraph_placekey']))

  # Filter main dataset for NYC supermarkets
  df = df.where(F.col('placekey').isin(set_supermarkets))
  
  

  # STEP 2: FILTER FOR VISITS OVERLAPPING WITH 4 MONTHS OF INTEREST

  # Extract months from date_range_start and date_range_end
  df = df.withColumn('start',df.date_range_start.substr(1,7))
  df = df.withColumn('end',df.date_range_end.substr(1,7))

  # Filter dataset for either start or end overlapping with 4 months of interest
  dates = {'2019-03','2019-10','2020-03','2020-10'}
  df = df.where(F.col('start').isin(dates) | F.col('end').isin(dates))

  # Create udf to retain only one date from both columns
  udf_dates = F.udf(lambda x,y: x if x in dates else y)
  df = df.withColumn('date', udf_dates(F.col('start'),F.col('end'))) \
    .select('placekey','poi_cbg','visitor_home_cbgs','date') \
    .cache()
  
  

  # STEP 3A: EXPLODE JSON INSIDE visitor_home_cbgs COLUMN TO MULTIPLE ROWS

  # Using method from FAQ section of Safegraph Docs

  def parser(element):
    return json.loads(element)

  jsonudf = F.udf(parser, MapType(StringType(), IntegerType()))

  visitor_home_cbgs_parsed = df.withColumn('parsed_visitor_home_cbgs', jsonudf('visitor_home_cbgs'))
  visitor_home_cbgs_exploded = visitor_home_cbgs_parsed.select('placekey','poi_cbg', 'date', F.explode('parsed_visitor_home_cbgs'))


  # STEP 3B: FILTER CENSUS BLOCK FIPS FOR NYC CENSUS BLOCKS USING nyc_cbg_centroids.csv

  # Read nyc_cbg_centroids.csv and cast latitude/longitude to floats
  df_cbgs = spark.read.csv('nyc_cbg_centroids.csv', header=True).cache()

  df_cbgs = df_cbgs.select('cbg_fips',
        df_cbgs['latitude'].cast('float'),
        df_cbgs['longitude'].cast('float'))

  # Create set of NYC census blocks
  set_cbgs = set(list(df_cbgs.select('cbg_fips').toPandas()['cbg_fips']))

  # Filter main dataset for NYC census blocks
  df2 = visitor_home_cbgs_exploded.where(F.col('key').isin(set_cbgs)).cache()



  # STEP 4A: REPROJECT COORDINATES TO EPSG 2263

  # Define function for transforming to 2263
  def projection_2263(lat, long):
    t = Transformer.from_crs(4326,2263)
    reprojected_x = t.transform(lat, long)[0]
    reprojected_y = t.transform(lat, long)[1]
    return (reprojected_x, reprojected_y)

  # Create udfs for getting x,y coordinates in 2263 using defined function
  udf_x = F.udf(lambda x,y: (projection_2263(x,y))[0])
  udf_y = F.udf(lambda x,y: (projection_2263(x,y))[1])

  # Apply udfs to cbg and supermarket coordinates
  df_cbgs = df_cbgs.withColumn('visitor_x', udf_x(F.col('latitude'),F.col('longitude'))).cache()
  df_cbgs = df_cbgs.withColumn('visitor_y', udf_y(F.col('latitude'),F.col('longitude'))).cache()

  df_supermarkets = df_supermarkets.withColumn('supermarkets_x', udf_x(F.col('latitude'),F.col('longitude'))).cache()
  df_supermarkets = df_supermarkets.withColumn('supermarkets_y', udf_y(F.col('latitude'),F.col('longitude'))).cache()



  # STEP 4B: JOIN ALL 3 DATAFRAMES

  # Join main dataframe with df_supermarkets on placekeys
  df3 = df2.join(df_supermarkets, df2['placekey']==df_supermarkets['safegraph_placekey'], how='inner') \
    .cache()

  # Join main dataframe with df_cbgs on cbg_fips
  df4 = df3.join(df_cbgs, df3['key']==df_cbgs['cbg_fips'], how='inner').cache()

  df4 = df4.select('cbg_fips','date',
                  df4['value'].alias('visitor_count'),
                  df4['supermarkets_x'].cast('float'),
                  df4['supermarkets_y'].cast('float'),
                  df4['visitor_x'].cast('float'),
                  df4['visitor_y'].cast('float')) \
                  .cache()



  # STEP 4C: CALCULATE DISTANCES

  # Define udf for calculating distance
  udf_dist = F.udf(lambda x,y,a,b: Point(x,y).distance(Point((a,b)))/5280, FloatType())

  # Create new column of distances
  df5 = df4.withColumn('dist', udf_dist(F.col('supermarkets_x'), F.col('supermarkets_y'), F.col('visitor_x'), F.col('visitor_y'))) \
    .cache()
  
  

  # STEP 5A: COMPUTE AVERAGE DISTANCE TRAVELLED PER CBG USING TOTAL DISTANCES AND TOTAL VISITOR NUMBERS

  # EXTRA CREDIT 1: FIND MEDIAN USING PERCENTILE_APPROX

  df_median = df5.groupBy('cbg_fips','date') \
    .agg(F.percentile_approx('dist',0.5).alias('median_dist')) \
    .withColumn('median_dist2', F.round(F.col('median_dist'), 2)) \
    .cache()
  
  
  # STEP 5B: GET FINAL OUTPUT

  # Pivot df by date
  # Sort by cbg_fips
  pivotdf2 = df_median.groupBy('cbg_fips').pivot('date').max('median_dist2') \
    .sort('cbg_fips') \
    .cache()

  # Replace null with blanks
  pivotdf2 = pivotdf2.select( *[ F.when(F.col(column).isNull(),'').otherwise(F.col(column)).alias(column) for column in pivotdf2.columns]) \
    .cache()

  # SAVE AS CSV
  import itertools
  output_rdd = pivotdf2.rdd.map(lambda x: (x[0],x[1],x[2],x[3],x[4])).cache()
  header = output_rdd.first()
  output_rdd.mapPartitions(lambda x: itertools.chain([header],x)) \
    .saveAsTextFile(sys.argv[1] if len(sys.argv)>1 else 'output')

if __name__=="__main__":
  sc = SparkContext()
  main(sc)