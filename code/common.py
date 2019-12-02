#!/usr/bin/env python

import pandas  as pd
import numpy   as np
import seaborn as sb


from itertools                 import chain
from math                      import log, ceil
from time                      import sleep, time
from os.path                   import exists, dirname, abspath, join
from os                        import system

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg         import Vectors
from pyspark.ml.stat           import Summarizer, Correlation
from pyspark.ml.feature        import ChiSqSelector, StringIndexer, OneHotEncoderEstimator, VectorAssembler, Imputer, StandardScaler, FeatureHasher
from pyspark.ml                import Pipeline

from pyspark.sql               import SparkSession, SQLContext
from pyspark.sql.types         import StructType, StructField, StringType, FloatType
from pyspark.sql.functions     import countDistinct, col, when, monotonically_increasing_id

#    ---------------------------------------------------------------------------------------------------------------------------------

workingSet = {}

def logMessage(msg):
    
    print('-' * 80 + '\n' + msg + '\n' + '-' * 80 )

    with open(join(dirname(__file__), 'log.txt'), 'a') as out:
        out.write(msg + '\n')

  # sleep( 3 )

def setupSpark(workingSet, application = 'w261', memory = '240G'):

    start = time()
    
    logMessage(f'Starting Spark Initializing')
    
    workingSet['ss'] = SparkSession.builder \
                                   .appName(application) \
                                   .config('spark.driver.memory', memory) \
                                   .getOrCreate()
    workingSet['sc'] = workingSet['ss'].sparkContext
    workingSet['sq'] = SQLContext(workingSet['sc'])

    logMessage(f'Finished Spark Initializing in {time()-start:.3f} Seconds')
    
def importData(workingSet, location = '../data', clean = False):

    start  = time()
    prefix = f'{location}/criteo.parquet'
    train  = f'{location}/train.txt'

    df_o   = f'df.full'

    file   = f'{prefix}.{df_o}'

    logMessage(f'Starting Data Importing at {location}')
    
    if  clean:
        system(f'rm -rf {location}/criteo.parquet.*')
    
    if  not exists(file):

        ds = StructType([StructField(f'label'  ,  FloatType(), True)                      ] + \
                        [StructField(f'i{f:02}',  FloatType(), True) for f in range(1, 14)] + \
                        [StructField(f's{f:02}', StringType(), True) for f in range(1, 27)])

        df = workingSet['sq'].read.format('csv') \
                             .options(header = 'true', delimiter = '\t') \
                             .schema(ds) \
                             .load(train)

        df.write.parquet(file)

    df = workingSet['ss'].read.parquet(file)

    workingSet['df.full'     ] = df

    workingSet['num_features'] = [f'{c}'          for c in df.columns if 'i'         in c]
    workingSet['std_features'] = [f'{c}_standard' for c in df.columns if 'i'         in c]
    workingSet['cat_features'] = [f'{c}'          for c in df.columns if 's'         in c]
    workingSet['all_features'] = [f'{c}'          for c in df.columns if 'label' not in c]
    
    workingSet['prefix'      ] = f'{location}/criteo.parquet'
    
    logMessage(f'Finished Data Importing in {time()-start:.3f} Seconds')

def splitFrame(workingSet, ratios = [0.8, 0.1, 0.1]):

    start  = time()
    prefix = workingSet['prefix']

    logMessage(f'Starting Data Splitting at {ratios}')
    
    splits = {}

    splits['train'], splits['test'], splits['dev'] = workingSet['df.full'].randomSplit(ratios, seed = 2019)
    splits['toy']                                  = workingSet['df.full'].sample(fraction = 0.001, seed = 2019)
    
    for subset in ['train','test','dev', 'toy']:
        
        df_o  = f'df.{subset}'
        file  = f'{prefix}.{df_o}'

        if  not exists(file):
            splits[subset].write.parquet(file)

        workingSet[df_o] = workingSet['ss'].read.parquet(file)
    
    workingSet['ratios'] = ratios
    
    logMessage(f'Finished Data Splitting in {time()-start:.3f} Seconds')