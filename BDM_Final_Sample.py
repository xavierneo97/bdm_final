from pyspark import SparkContext
import itertools

if __name__=='__main__':
    sc = SparkContext()
    rdd = sc.textFile('/tmp/bdm/weekly-patterns-nyc-2019-2020/*')
    header = rdd.first()
    rdd.sample(False, 0.01) \
        .coalesce(1) \
        .mapPartitions(lambda x: itertools.chain([header], x)) \
        .saveAsTextFile('weekly-patterns-nyc-2019-2020-sample')