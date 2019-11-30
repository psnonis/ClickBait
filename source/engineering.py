from code.common               import *

def catFillUndefined(workingSet, subset = 'toy', term = 'deadbeef'):

    start = ti.time()
    file  = f'{workingSet["data"]}/criteo.parquet.{subset}.filled'

    log(f'Starting Categorical Fill Undefined Terms on {subset}')

    if  not exists(file):
        
        df = workingSet[f'df_{subset}']

        df = df.fillna(term, workingSet['cat_features'])

        df.write.parquet(file)

    workingSet[f'df_{subset}'] = workingSet['ss'].read.parquet(file)

    log(f'Finished Categorical Fill Undefined Terms in {ti.time()-start:.3f} Seconds')

def catFindFrequent(workingSet, subset = 'toy', threshold = 360000, remember = True):
    
    start = ti.time()

    log(f'Starting Categorical Find Frequent Terms on {subset}')

    df = workingSet[f'df_{subset}']

    distinct = {}
    frequent = {}
    uncommon = {}

    for feature in workingSet['cat_features'] :
        df_count          = df.select(feature).groupBy(feature).count()
        uncommon[feature] = df_count.filter(df_count['count'] <  threshold).sort('count', ascending = False).select(feature).rdd.flatMap(list).collect()
        frequent[feature] = df_count.filter(df_count['count'] >= threshold).sort('count', ascending = False).select(feature).rdd.flatMap(list).collect()
        distinct[feature] = uncommon[feature] + frequent[feature]

        print(feature, f'found {len(uncommon[feature]):>8} uncommon categories of {len(distinct[feature]):>8} distinct categories -> {len(frequent[feature]):>3} frequent categories = {frequent[feature]}')

    if  remember:

        workingSet['distinct' ] = distinct
        workingSet['frequent' ] = frequent
        workingSet['uncommon' ] = uncommon

        workingSet['threshold'] = threshold

    log(f'Finished Categorical Find Frequent Terms in {ti.time()-start:.3f} Seconds')

def catMaskUncommon(workingSet, subset = 'toy', term = 'rarebeef'):

    start = ti.time()
    file  = f'{workingSet["data"]}/criteo.parquet.{subset}.filled.frequent-{workingSet["threshold"]}'

    log(f'Starting Categorical Mask Uncommon Terms on {subset}')

    if  not exists(file):
        
        df = workingSet[f'df_{subset}']

        """
        for feature, uncommon_categories in workingSet['uncommon'].items():
            df = df.replace(uncommon_categories, term)
        """
        
        for feature, frequent_categories in workingSet['frequent'].items():
            df = df.withColumn(feature, when(~df[feature].isin(*frequent_categories), term).otherwise(df[feature]))

        df.write.parquet(file)

    workingSet[f'df_{subset}'] = workingSet['ss'].read.parquet(file)

    log(f'Finished Categorical Mask Uncommon Terms in {ti.time()-start:.3f} Seconds')

def hashFeatures(workingSet, subset = 'toy'):

    start = ti.time()
    file  = f'{workingSet["data"]}/criteo.parquet.{subset}.hashed'

    log(f'Starting Feature Hashing on {subset}')

    
    if  not exists(file):
        
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
