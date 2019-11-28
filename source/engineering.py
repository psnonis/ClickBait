from common import *

def categoricalFillNA(workingSet, subset = 'toy', term = 'deadbeef'):
    
    start = ti.time()

    log(f'Starting Categorical FillNA Hashing on {subset}')

    df.fillna('deadbeef', workingSet['cat_features'])

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
