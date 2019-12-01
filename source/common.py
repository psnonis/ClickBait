#!/usr/bin/env python

import pandas  as pd
import numpy   as np
import seaborn as sb

from math                      import log, ceil
from time                      import sleep, time
from os.path                   import exists, dirname, abspath, join
from os                        import system

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg         import Vectors
from pyspark.ml.feature        import ChiSqSelector, StringIndexer, OneHotEncoderEstimator, VectorAssembler, Imputer, StandardScaler, FeatureHasher
from pyspark.ml                import Pipeline

from pyspark.sql               import SparkSession, SQLContext
from pyspark.sql.types         import StructType, StructField, StringType, FloatType
from pyspark.sql.functions     import countDistinct, col, when

#    ---------------------------------------------------------------------------------------------------------------------------------

workingSet = {}

def logMessage(msg):
    
    print('-' * 80 + '\n' + msg + '\n' + '-' * 80 )

    with open(join(dirname(__file__), 'log.txt'), 'a') as out:
        out.write(msg + '\n')

  # sleep( 3 )

def initSpark(workingSet, application = 'w261', memory = '240G'):

    start = time()
    
    logMessage(f'Starting Spark Initializing')
    
    workingSet['ss'] = SparkSession.builder \
                                   .appName(application) \
                                   .config('spark.driver.memory', memory) \
                                   .getOrCreate()
    workingSet['sc'] = workingSet['ss'].sparkContext
    workingSet['sq'] = SQLContext(workingSet['sc'])

    logMessage(f'Finished Spark Initializing in {time()-start:.3f} Seconds')
    
def loadData(workingSet, data = '../data', clean = False):

    start = time()
    data  = abspath(data)
    file  = f'{data}/criteo.parquet.full'
    train = f'{data}/train.txt'
    
    logMessage(f'Starting Data Loading at {data}')
    
    if  clean:
        system(f'rm -rf {data}/criteo.parquet.*')
    
    if  not exists(file):

        ds = StructType([StructField(f'ctr'    ,  FloatType(), True)                      ] + \
                        [StructField(f'i{f:02}',  FloatType(), True) for f in range(1, 14)] + \
                        [StructField(f's{f:02}', StringType(), True) for f in range(1, 27)])

        df = workingSet['sq'].read.format('csv') \
                             .options(header = 'true', delimiter = '\t') \
                             .schema(ds) \
                             .load(train)

        df.write.parquet(file)

    df = workingSet['ss'].read.parquet(file)

    workingSet['df_full'     ] = df
    workingSet['df_toy'      ] = df.sample(fraction = 0.001, seed = 2019)

    workingSet['num_features'] = [c for c in df.columns if 'i'       in c]
    workingSet['cat_features'] = [c for c in df.columns if 's'       in c]
    workingSet['all_features'] = [c for c in df.columns if 'ctr' not in c]
    
    workingSet['data'        ] = data
    
    logMessage(f'Finished Data Loading in {time()-start:.3f} Seconds')

def splitData(workingSet, ratios = [0.8, 0.1, 0.1]):

    start = time()

    logMessage(f'Starting Data Splitting at {ratios}')
    
    splits = {}

    splits['train'], splits['test'], splits['dev'] = workingSet['df_full'].randomSplit(ratios, seed = 2019)
    
    for subset in ['train','test','dev']:
        
        file = f'{workingSet["data"]}/criteo.parquet.{subset}'
        
        if  not exists(file):
            splits[subset].write.parquet(file)

        workingSet[f'df_{subset}'] = workingSet['ss'].read.parquet(file)
    
    workingSet['ratios'] = ratios
    
    logMessage(f'Finished Data Splitting in {time()-start:.3f} Seconds')