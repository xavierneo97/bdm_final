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

  spark = SparkSession(sc)

  

  def reproject(x, y):
    t = Transformer.from_crs(4326,2263)
    reprojected = t.transform(x,y)
    new_x = str(float(reprojected[0]))
    new_y = str(float(reprojected[1]))
    return str(new_x+'-'+new_y) 

  def obtain_distance(x, y, a, b):
    dist = Point(x,y).distance(Point((a,b)))/5280
    return round(dist, 2)

  # Create udf
  udf_proj = F.udf(reproject)
  udf_dist = F.udf(obtain_distance)


 

  # Read nyc_supermarkets.csv and take columns of interest: safegraph_placekey (to match to main dataset), latitude and longitude
  df_supermarkets = spark.read.csv('nyc_supermarkets.csv', header=True) \
    .select('safegraph_placekey',
          F.col('latitude').alias('supermarkets_x').cast('float'),
          F.col('longitude').alias('supermarkets_y').cast('float')) \
          .withColumn('reprojected', udf_proj(F.col('supermarkets_x'),F.col('supermarkets_y'))) \
          .withColumn('supermarkets_x', F.split(F.col('reprojected'), '-')[0].cast(FloatType())) \
          .withColumn('supermarkets_y', F.split(F.col('reprojected'), '-')[1].cast(FloatType())) \
          .cache()

  set_supermarkets = set(df_supermarkets.select('safegraph_placekey').distinct().rdd.flatMap(lambda x: x).collect())

  dates = {'2019-03','2019-10','2020-03','2020-10'}
  # Create udf to retain only one date from both columns
  udf_dates = F.udf(lambda x,y: x if x in dates else y)

  import json

  def parser(element):
    return json.loads(element)

  jsonudf = F.udf(parser, MapType(StringType(), IntegerType()))



  df = spark.read.csv('weekly-patterns-nyc-2019-2020-sample.csv',
                      header=True,
                      escape='\"') \
                      .select('placekey','poi_cbg','visitor_home_cbgs','date_range_start','date_range_end') \
                      .where(F.col('placekey').isin(set_supermarkets)) \
                      .withColumn('start',F.col('date_range_start').substr(1,7)) \
    .withColumn('end',F.col('date_range_end').substr(1,7)) \
    .where(F.col('start').isin(dates) | F.col('end').isin(dates)) \
    .withColumn('date', udf_dates(F.col('start'),F.col('end'))) \
    .select('placekey','poi_cbg','visitor_home_cbgs','date') \
    .withColumn('parsed_visitor_home_cbgs', jsonudf('visitor_home_cbgs')) \
    .select('placekey','poi_cbg', 'date', F.explode('parsed_visitor_home_cbgs')) \
    .cache()

  # Inner join and filter main dataframe with df_supermarkets on placekeys
  df2 = df.join(df_supermarkets, df['placekey']==df_supermarkets['safegraph_placekey'], how='inner') \
    .cache()

  set_cbgs = set(df2.select('key').distinct().rdd.flatMap(lambda x: x).collect())

 
  def get_avg(x,y):
    return round(x/y, 2)

  udf_avg = F.udf(get_avg)

  # Read nyc_cbg_centroids.csv
  df_cbgs = spark.read.csv('nyc_cbg_centroids.csv', header=True) \
    .select('cbg_fips',
          F.col('latitude').alias('cbg_x').cast('float'),
          F.col('longitude').alias('cbg_y').cast('float')) \
          .where(F.col('cbg_fips').isin(set_cbgs)) \
          .withColumn('reprojected', udf_proj(F.col('cbg_x'),F.col('cbg_y'))) \
          .withColumn('cbg_x', F.split(F.col('reprojected'), '-')[0].cast(FloatType())) \
          .withColumn('cbg_y', F.split(F.col('reprojected'), '-')[1].cast(FloatType())) \
          .cache()

  # Join main dataframe with df_cbgs on cbg_fips
  df4 = df2.join(df_cbgs, df2['key']==df_cbgs['cbg_fips'], how='inner') \
    .select('cbg_fips','date',
                  F.col('value').alias('visitor_count'),
                  'supermarkets_x','supermarkets_y','cbg_x','cbg_y') \
                  .withColumn('dist', udf_dist(F.col('supermarkets_x'), F.col('supermarkets_y'), F.col('cbg_x'), F.col('cbg_y'))) \
    .select('cbg_fips','date','visitor_count',
            F.col('dist').cast(FloatType())) \
    .groupBy('cbg_fips','date').sum('visitor_count','dist') \
    .withColumn('avg_dist', udf_avg(F.col('sum(dist)'),F.col('sum(visitor_count)'))) \
    .withColumn('avg_dist2', F.col('avg_dist').cast(FloatType())) \
    .groupBy('cbg_fips').pivot('date').max('avg_dist2') \
    .sort('cbg_fips') \
    .write.option('header','true') \
    .csv(sys.argv[1] if len(sys.argv)>1 else 'output')

if __name__=="__main__":
  sc = SparkContext()
  main(sc)


