#!/usr/bin/env python

import pandas  as pd
import numpy   as np
import seaborn as sb
import time    as ti

from os.path                   import exists

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg         import Vectors
from pyspark.ml.feature        import ChiSqSelector, StringIndexer, VectorAssembler, Imputer, StandardScaler, FeatureHasher
from pyspark.ml                import Pipeline

from pyspark.sql               import SparkSession, SQLContext
from pyspark.sql.types         import StructType, StructField, StringType, FloatType
from pyspark.sql.functions     import countDistinct, col

#    ---------------------------------------------------------------------------------------------------------------------------------

def log(msg):
    
    print('\n' + '-' * 80 + '\n' + msg + '\n' + '-' * 80 )

    with open('lr.out', 'a') as out:
        out.write(msg)
        
    ti.sleep( 3 )

def initSpark(workingSet):

    start = ti.time()
    
    log(f'Starting Spark Initializing')
    
    workingSet['ss'] = SparkSession.builder \
                                   .config('spark.driver.memory', '240G') \
                                   .getOrCreate()
    workingSet['sc'] = workingSet['ss'].sparkContext
    workingSet['sq'] = SQLContext(workingSet['sc'])

    print(f'Finished Spark Initializing in {ti.time()-start:.3f} Seconds')
    
def loadData(workingSet):

    start = ti.time()
    
    log(f'Starting Data Loading')
    
    if  not exists('../data/criteo.parquet.full'):

        ds = StructType([StructField(f'ctr'    ,  FloatType(), True)                      ] + \
                        [StructField(f'i{f:02}',  FloatType(), True) for f in range(1, 14)] + \
                        [StructField(f's{f:02}', StringType(), True) for f in range(1, 27)])

        df = workingSet['sq'].read.format('csv') \
                             .options(header = 'true', delimiter = '\t') \
                             .schema(ds) \
                             .load('../data/train.txt')

        df.write.parquet('../data/criteo.parquet.full')

    df = workingSet['ss'].read.parquet('../data/criteo.parquet.full')

    workingSet['df_full'    ] = df
    workingSet['df_toy'     ] = df.sample(fraction = 0.001, seed = 2019)

    workingSet['num_columns'] = [c for c in df.columns if 'i'       in c]
    workingSet['cat_columns'] = [c for c in df.columns if 's'       in c]
    workingSet['all_columns'] = [c for c in df.columns if 'ctr' not in c]
    
    print(f'Finished Data Loading in {ti.time()-start:.3f} Seconds')

def splitData(workingSet):

    start = ti.time()

    log(f'Starting Data Splitting')
    
    if  not exists('../data/criteo.parquet.train') or \
        not exists('../data/criteo.parquet.test' ) or \
        not exists('../data/criteo.parquet.dev'  )    :

        train, test, dev = workingSet['df_full'].randomSplit([0.8, 0.1, 0.1], seed = 2019)
        
        train.write.parquet('../data/criteo.parquet.train')
        test.write.parquet('../data/criteo.parquet.test')
        dev.write.parquet('../data/criteo.parquet.dev')
        
    workingSet['df_train'] = workingSet['ss'].read.parquet('../data/criteo.parquet.train')
    workingSet['df_test '] = workingSet['ss'].read.parquet('../data/criteo.parquet.test')
    workingSet['df_dev'  ] = workingSet['ss'].read.parquet('../data/criteo.parquet.dev')
    
    log(f'Finished Data Splitting in {ti.time()-start:.3f} Seconds')
    
def hashFeatures(workingSet, subset = 'toy'):

    start = ti.time()

    log(f'Starting Feature Hashing on {subset}')
    
    if  not exists(f'../data/criteo.parquet.{subset}.hashed'):
        
        features = 33554432
        hasher   = FeatureHasher(inputCols = workingSet['all_columns'], outputCol = 'features', numFeatures = features )

        df       = hasher.transform(workingSet[f'df_{subset}'])
        df       = df.select(['ctr', 'features']).withColumnRenamed('ctr','label')

        df.write.parquet(f'../data/criteo.parquet.{subset}.hashed')
      # df.write.partitionBy('label').parquet(f'../data/criteo.parquet.{subset}.hashed')

    workingSet[f'df_{subset}_hashed'] = workingSet['ss'].read.parquet(f'../data/criteo.parquet.{subset}.hashed')

    log(f'Finished Feature Hashing in {ti.time()-start:.3f} Seconds')

def selectFeatures(workingSet, subset = 'toy'):
    
    start = ti.time()
    
    log(f'Starting Feature Selection on {subset}')
    
    if  not exists(f'../data/criteo.parquet.{subset}.selected'):
        
        features = 1000000
        selector = ChiSqSelector(numTopFeatures = features, featuresCol = 'features', outputCol = 'selectedFeatures', labelCol = 'label')

        df       = workingSet[f'df_{subset}_hashed']
        df       = selector.fit(df).transform(df)
        df       = df.select(['label', 'selectedFeatures']).withColumnRenamed('selectedFeatures','features')
        
        df.write.parquet(f'../data/criteo.parquet.{subset}.selected')
      # df.write.partitionBy('label').parquet(f'../data/criteo.parquet.{subset}.selected')

    workingSet[f'df_{subset}_selected'] = workingSet['ss'].read.parquet(f'../data/criteo.parquet.{subset}.selected')

    log(f'Finished Feature Selection in {ti.time()-start:.3f} Seconds')
    
def trainLR(workingSet, subset = 'toy'):

    log(f'Starting LR Training on {subset}')
    
    start = ti.time()

    df    = workingSet[f'df_{subset}_selected']
    lr    = LogisticRegression(maxIter = 10, regParam = 0.3, elasticNetParam = 0.8)
    model = lr.fit(df)
    
    workingSet['lr'   ] = lr
    workingSet['model'] = model

    log(f'\nFinished LR Training in {ti.time()-start:.3f} Seconds\n')
    
def main():    

    log(f'\nStarting\n')
    
    workingSet = {}

    initSpark(workingSet)
    
    loadData(workingSet)

  # hashFeatures(workingSet, 'train')
  # hashFeatures(workingSet, 'test')
  # hashFeatures(workingSet, 'dev')
    hashFeatures(workingSet, 'toy')

    selectFeatures(workingSet, 'toy')

    trainLR(workingSet, 'toy')

    log(f'\nStopping\n')
    
if  __name__ == '__main__':
    main()