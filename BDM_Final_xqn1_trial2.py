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
  import pandas as pd
  import numpy as np
  from pyproj import Transformer
  from shapely.geometry import Point
  from itertools import chain

  spark = SparkSession(sc)

  # STEP 1: FILTER MAIN DATAFRAME WITH nyc_supermarkets.csv

  def reproject(x, y):
    t = Transformer.from_crs(4326,2263)
    reprojected = t.transform(x,y)
    new_x = str(float(reprojected[0]))
    new_y = str(float(reprojected[1]))
    return str(new_x+'_'+new_y) 

  def obtain_distance(x, y, a, b):
    dist = Point(x,y).distance(Point((a,b)))/5280
    return round(dist, 2)

  # Create udf
  udf_proj = F.udf(reproject)
  udf_dist = F.udf(obtain_distance)

  # Read nyc_supermarkets.csv and take columns of interest: safegraph_placekey (to match to main dataset), latitude and longitude
  df_supermarkets = spark.read.csv('nyc_supermarkets.csv', header=True) \
    .select('safegraph_placekey', 
            F.col('latitude').cast(FloatType()), F.col('longitude').cast(FloatType())) \
                    .na.drop() \
    .withColumn('reproject', udf_proj(F.col('latitude'), F.col('longitude'))) \
    .withColumn('supermarkets_x', F.split(F.col('reproject'),'_').getItem(0).cast(FloatType())) \
    .withColumn('supermarkets_y', F.split(F.col('reproject'),'_').getItem(1).cast(FloatType())) \
    .select('safegraph_placekey','supermarkets_x','supermarkets_y') \
          .cache()

  set_supermarkets = set(df_supermarkets.select('safegraph_placekey').distinct().rdd.flatMap(lambda x: x).collect())

  # Filter main dataframe with df_supermarkets on placekeys

  df = spark.read.csv('/tmp/bdm/weekly-patterns-nyc-2019-2020/*',
                      header=True,
                      escape='\"') \
                      .select('placekey','poi_cbg','visitor_home_cbgs','date_range_start','date_range_end') \
                      .na.drop() \
                      .where(F.col('placekey').isin(set_supermarkets)) \
                      .cache()



  # STEP 2: FILTER FOR VISITS OVERLAPPING WITH 4 MONTHS OF INTEREST

  # Filter dataset for either start or end overlapping with 4 months of interest
  dates = {'2019-03','2019-10','2020-03','2020-10'}

  # Create udf to retain only one date from both columns
  udf_dates = F.udf(lambda x,y: x if x in dates else y)

  # Extract months from date_range_start and date_range_end
  df2 = df.withColumn('start',df.date_range_start.substr(1,7)) \
    .withColumn('end',df.date_range_end.substr(1,7)) \
    .where(F.col('start').isin(dates) | F.col('end').isin(dates)) \
    .withColumn('date', udf_dates(F.col('start'),F.col('end'))) \
    .select('placekey','poi_cbg','visitor_home_cbgs','date') \
    .cache()



  # STEP 3: FILTER FOR VISITS IN nyc_cbg_centroids.csv

  # Read nyc_cbg_centroids.csv
  df_cbgs = spark.read.csv('nyc_cbg_centroids.csv', header=True) \
    .select('cbg_fips',
          F.col('latitude').cast('float'),
          F.col('longitude').cast('float')) \
    .na.drop() \
    .withColumn('reproject', udf_proj(F.col('latitude'), F.col('longitude'))) \
    .withColumn('cbg_x', F.split(F.col('reproject'),'_').getItem(0).cast(FloatType())) \
    .withColumn('cbg_y', F.split(F.col('reproject'),'_').getItem(1).cast(FloatType())) \
    .select('cbg_fips','cbg_x','cbg_y') \
    .cache()

  set_cbgs = set(df_cbgs.select('cbg_fips').distinct().rdd.flatMap(lambda x: x).collect())


  # Explode json inside visitor_home_cbgs column into multiple rows
  # Using method from FAQ section of Safegraph Docs
  # Filter visitor_home_cbgs with nyc_cbg_centroids

  import json

  def parser(element):
    return json.loads(element)

  jsonudf = F.udf(parser, MapType(StringType(), IntegerType()))

  visitor_home_cbgs_parsed = df2.withColumn('parsed_visitor_home_cbgs', jsonudf('visitor_home_cbgs')).cache()
  df3 = visitor_home_cbgs_parsed.select('placekey','poi_cbg', 'date', F.explode('parsed_visitor_home_cbgs')) \
    .groupBy('placekey','poi_cbg','date','key').sum('value') \
    .where(F.col('key').isin(set_cbgs)) \
    .cache()




  # STEP 4: CREATE A DF OF UNIQUE ORIGIN-DESTINATION PAIRS TO CALCULATE DISTANCES ONCE PER PAIR

  # Concat origin-destination columns
  def concat(x,y):
    o_d = str(x)+'_'+str(y)
    return str(o_d)

  merge_od = F.udf(concat)

  # Calculate distances using df of unique origin-destination pairs 

  dist_df = df3.withColumn('origin-dest', merge_od(F.col('placekey'), F.col('key'))) \
    .select('origin-dest', 'placekey', 'key') \
    .distinct() \
    .join(df_supermarkets, F.col('placekey')==df_supermarkets['safegraph_placekey'], how='inner') \
    .join(df_cbgs, F.col('key')==df_cbgs['cbg_fips'], how='inner') \
    .withColumn('dist', udf_dist(F.col('supermarkets_x'), F.col('supermarkets_y'), F.col('cbg_x'), F.col('cbg_y'))) \
    .select('origin-dest','dist') \
    .cache()

  # Create dictionary of distances per origin-destination pair
  # Use dictionary to map original df

  dist_dict = {x['origin-dest']: x['dist'] for x in dist_df.collect()}
  mapping_expr = F.create_map([F.lit(x) for x in chain(*dist_dict.items())])



  # STEP 5: GET AVERAGE DISTANCE AND FINAL OUTPUT

  # Get total visitor count and distance per cbg
  # Compute average distance per cbg
  # Pivot by date
  # Sort by cbg_fips

  df4 = df3.withColumn('origin-dest', merge_od(F.col('placekey'), F.col('key'))) \
    .withColumn('dist', mapping_expr.getItem(F.col('origin-dest'))) \
    .na.drop() \
    .select(F.col('key').alias('cbg_fips'),
            'date',
            F.col('sum(value)').alias('visitor_count'),
            F.col('dist').cast(FloatType())) \
    .groupBy('cbg_fips','date').sum('visitor_count','dist') \
    .withColumn('avg_dist', F.round(F.col('sum(dist)')/F.col('sum(visitor_count)'), 2)) \
    .groupBy('cbg_fips').pivot('date').max('avg_dist') \
    .sort('cbg_fips') \
    .write.option('header','true') \
    .csv(sys.argv[1] if len(sys.argv)>1 else 'output')


if __name__=="__main__":
  sc = SparkContext()
  main(sc)


