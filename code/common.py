import pandas  as pd
import numpy   as np
import seaborn as sb

from itertools                 import chain
from math                      import log, ceil
from time                      import sleep, time
from os.path                   import exists, dirname, abspath, join
from os                        import system
from glob                      import glob
from pickle                    import dump, load
from datetime                  import datetime

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg         import Vectors
from pyspark.ml.stat           import Summarizer, Correlation
from pyspark.ml.feature        import ChiSqSelector, StringIndexer, OneHotEncoderEstimator, VectorAssembler, Imputer, StandardScaler, FeatureHasher
from pyspark.ml                import Pipeline, PipelineModel

from pyspark.sql               import SparkSession, DataFrame
from pyspark.sql.types         import StructType, StructField, IntegerType, StringType, FloatType
from pyspark.sql.functions     import countDistinct, col, when, rand

from typing                    import List, Dict, Tuple, Any

def timePrint(message: str):
    
    timeStamp = datetime.now().strftime('%H:%M:%S')
    output    = f'[{timeStamp}] : {message}'
    
    print(output)
    
    with open('messages.txt', 'a') as log:
        log.write(output + '\n')

class Common(object):

    prefix = 'data'
    spark  = None

    num_features = [f'n{f:02d}'          for f in range(1, 13+1)]
    std_features = [f'n{f:02d}_standard' for f in range(1, 13+1)]
    cat_features = [f'c{f:02d}'          for f in range(1, 26+1)]
    
    cat_uncommon = {}
    cat_frequent = {}
    cat_distinct = {}

    num_measures = {}
    cat_measures = {}

    def setupSpark(application = 'w261', memory = '220G'):

        timePrint('Starting Spark Initialization')

        Common.spark = SparkSession.builder \
            .appName(application) \
            .config('spark.driver.memory', memory) \
            .getOrCreate()

        timePrint('Stopping Spark Initialization\n')

    def importData(location: str = 'data', clean: bool = False):

        timePrint('Starting Data Import')
        
        Common.prefix = f'{location}'
        
        frame = f'whole'
        train = f'{location}/{frame}.zip'
        whole = f'{location}/{frame}.parquet'

        if  clean:

            system(f'rm -rf {location}/train.parquet.*')
            system(f'rm -rf {location}/tests.parquet.*')
            system(f'rm -rf {location}/valid.parquet.*')

            system(f'rm -rf {location}/model.pickled.*')

        if  not exists(whole):

            schema = StructType([StructField(f'label', IntegerType(), True)                             ] + \
                                [StructField(f'{f}',     FloatType(), True) for f in Common.num_features] + \
                                [StructField(f'{f}',    StringType(), True) for f in Common.cat_features])

            criteo = Common.spark.read.format('csv') \
                .options(header = 'false', delimiter = '\t') \
                .schema(schema) \
                .load(train)

            criteo.write.parquet(whole)

        timePrint('Stopping Data Import\n')

    def splitsData(ratios = [0.8, 0.1, 0.1]):

        timePrint("Starting Data Splits")

        splits = {}
        whole  = Common.spark.read.parquet(f'{Common.prefix}/whole.parquet')

        splits['train'], splits['tests'], splits['valid'] = whole.randomSplit(ratios, seed = 2019)

        for subset in ['train','tests','valid']:

            path  = f'{Common.prefix}/{subset}.parquet'

            if  not exists(path):
                splits[subset].write.parquet(path)

        timePrint('Stopping Data Splits\n')

    def imp(subset, step):
        
        df = Common.spark.read.parquet(f'data/{subset}.parquet.{step}')
        
        return df

    def pdf(path, rows = 20, filter = ''):
        
        df = Common.spark.read.parquet(path)
        
        if  filter:
            df = df.filter(filter)

        return pd.DataFrame(df.take(rows), columns = df.columns)