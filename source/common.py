#!/usr/bin/env python

import pandas  as pd
import numpy   as np
import seaborn as sb
import time    as ti

from os.path                   import exists, dirname, join

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg         import Vectors
from pyspark.ml.feature        import ChiSqSelector, StringIndexer, VectorAssembler, Imputer, StandardScaler, FeatureHasher
from pyspark.ml                import Pipeline

from pyspark.sql               import SparkSession, SQLContext
from pyspark.sql.types         import StructType, StructField, StringType, FloatType
from pyspark.sql.functions     import countDistinct, col

#    ---------------------------------------------------------------------------------------------------------------------------------

workingSet = {}

def log(msg):
    
    print('\n' + '-' * 80 + '\n' + msg + '\n' + '-' * 80 )

    with open(join(dirname(__file__), 'log.txt'), 'a') as out:
        out.write(msg + '\n')

    ti.sleep( 3 )

def initSpark(workingSet, application = 'w261', memory = '240G'):

    start = ti.time()
    
    log(f'Starting Spark Initializing')
    
    workingSet['ss'] = SparkSession.builder \
                                   .appName(application) \
                                   .config('spark.driver.memory', memory) \
                                   .getOrCreate()
    workingSet['sc'] = workingSet['ss'].sparkContext
    workingSet['sq'] = SQLContext(workingSet['sc'])

    log(f'Finished Spark Initializing in {ti.time()-start:.3f} Seconds')
    
def loadData(workingSet, location = '../data'):

    start = ti.time()
    
    log(f'Starting Data Loading')
    
    if  not exists(f'{location}/criteo.parquet.full'):

        ds = StructType([StructField(f'ctr'    ,  FloatType(), True)                      ] + \
                        [StructField(f'i{f:02}',  FloatType(), True) for f in range(1, 14)] + \
                        [StructField(f's{f:02}', StringType(), True) for f in range(1, 27)])

        df = workingSet['sq'].read.format('csv') \
                             .options(header = 'true', delimiter = '\t') \
                             .schema(ds) \
                             .load(f'{location}/train.txt')

        df.write.parquet(f'{location}/criteo.parquet.full')

    df = workingSet['ss'].read.parquet(f'{location}/criteo.parquet.full')

    workingSet['df_full'     ] = df
    workingSet['df_toy'      ] = df.sample(fraction = 0.001, seed = 2019)

    workingSet['num_features'] = [c for c in df.columns if 'i'       in c]
    workingSet['cat_features'] = [c for c in df.columns if 's'       in c]
    workingSet['all_features'] = [c for c in df.columns if 'ctr' not in c]
    
    log(f'Finished Data Loading in {ti.time()-start:.3f} Seconds')

def splitData(workingSet, location = '../data', ratios = [0.8, 0.1, 0.1]):

    start = ti.time()

    log(f'Starting Data Splitting')
    
    if  not exists(f'{location}/criteo.parquet.train') or \
        not exists(f'{location}/criteo.parquet.test' ) or \
        not exists(f'{location}/criteo.parquet.dev'  )    :

        train, test, dev = workingSet['df_full'].randomSplit(ratios, seed = 2019)
        
        train.write.parquet(f'{location}/criteo.parquet.train')
        test.write.parquet(f'{location}/criteo.parquet.test')
        dev.write.parquet(f'{location}/criteo.parquet.dev')
        
    workingSet['df_train'] = workingSet['ss'].read.parquet(f'{location}/criteo.parquet.train')
    workingSet['df_test '] = workingSet['ss'].read.parquet(f'{location}/criteo.parquet.test')
    workingSet['df_dev'  ] = workingSet['ss'].read.parquet(f'{location}/criteo.parquet.dev')
    
    log(f'Finished Data Splitting in {ti.time()-start:.3f} Seconds')