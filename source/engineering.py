from code.common import *

### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def catFillUndefined(workingSet, subset = 'toy', term = 'deadbeef'):

    start = time()

    file  = f'{workingSet["data"]}/criteo.parquet.{subset}.filled'
    df_i  = f'df_{subset}'
    df_o  = f'df_{subset}'

    logMessage(f'Starting Categorical Fill Undefined Terms on {subset}')

    if  not exists(file):
        
        df = workingSet[df_i]

        df = df.fillna(term, workingSet['cat_features'])

        df.write.parquet(file)

    workingSet[df_o] = workingSet['ss'].read.parquet(file)

    logMessage(f'Finished Categorical Fill Undefined Terms in {time()-start:.1f} Seconds')

### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def catFindFrequent(workingSet, subset = 'toy', threshold = 180000, fit = False):
    
    start = time()

    logMessage(f'Starting Categorical Find Frequent Terms on {subset}')

    if  fit:
        df = workingSet[f'df_{subset}']
    else:
        df = workingSet[f'df_{subset}_{threshold}']

    distinct = {}
    frequent = {}
    uncommon = {}
    
    total_distinct = 0
    total_frequent = 0
    total_uncommon = 0
    
    for feature in workingSet['cat_features'] :
        df_count          = df.select(feature).groupBy(feature).count()
        uncommon[feature] = df_count.filter(df_count['count'] <  threshold).sort('count', ascending = False).select(feature).rdd.flatMap(list).collect()
        frequent[feature] = df_count.filter(df_count['count'] >= threshold).sort('count', ascending = False).select(feature).rdd.flatMap(list).collect()
        distinct[feature] = uncommon[feature] + frequent[feature]
        
        total_uncommon   += len(uncommon[feature])
        total_frequent   += len(frequent[feature])
        total_distinct   += len(distinct[feature])

        print(f'{feature} found {len(uncommon[feature]):>8} uncommon categories of {len(distinct[feature]):>8} distinct categories -> {len(frequent[feature]):>3} frequent categories = {" ".join(frequent[feature])}')

    print(f'\nall found {total_uncommon:>8} uncommon categories of {total_distinct:>8} distinct categories -> {total_frequent:>3} frequent categories')
        
    if  fit:

        workingSet['distinct' ] = distinct
        workingSet['frequent' ] = frequent
        workingSet['uncommon' ] = uncommon

        workingSet['threshold'] = threshold
        
    logMessage(f'Finished Categorical Find Frequent Terms in {time()-start:.1f} Seconds')

### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def catMaskUncommon(workingSet, subset = 'toy', threshold = 180000, term = 'rarebeef'):

    start = time()

    file  = f'{workingSet["data"]}/criteo.parquet.{subset}.filled.masked-{threshold}'
    df_i  = f'df_{subset}'
    df_o  = f'df_{subset}_{threshold}'

    logMessage(f'Starting Categorical Mask Uncommon Terms on {subset}')

    if  not exists(file):
        
        df = workingSet[df_i]

        """
        for feature, uncommon_categories in workingSet['uncommon'].items():
            df = df.replace(uncommon_categories, term)
        """

        for feature, frequent_categories in workingSet['frequent'].items():
            df = df.withColumn(feature, when(~df[feature].isin(*frequent_categories), term).otherwise(df[feature]))

        df.write.parquet(file)

    workingSet[df_o] = workingSet['ss'].read.parquet(file)

    logMessage(f'Finished Categorical Mask Uncommon Terms in {time()-start:.1f} Seconds')

### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def catCodeFeatures(workingSet, subset = 'toy', threshold = 180000, fit = False):

    start = time()

    file  = f'{workingSet["data"]}/criteo.parquet.{subset}.filled.masked-{threshold}.encode'
    df_i  = f'df_{subset}_{threshold}'
    df_o  = f'df_{subset}_{threshold}_encode'

    logMessage(f'Starting Categorical Feature Encoding on {subset}')
    
    df     = workingSet[df_i]

    if  fit:

        stages   = []

        features = workingSet['cat_features']
        indexes  = [f'{f}_index'  for f in features]
        vectors  =[ f'{f}_vector' for f in features]

        for feature, index, vector in zip(features, indexes, vectors):
            indexer  = StringIndexer(inputCol = feature, outputCol = index)
            encoder  = OneHotEncoderEstimator(inputCols = [indexer.getOutputCol()], outputCols = [vector], dropLast = False) # handleInvalid = 'keep'
            stages  += [indexer, encoder]

        assembler = VectorAssembler(inputCols = vectors, outputCol = 'features')
        stages   += [assembler]

        pipeline  = Pipeline(stages = stages)
        model     = pipeline.fit(df)
                       
        workingSet['code_model'] = model
                       
    else:

        model = workingSet['code_model']
    
    if  not exists(file):
        
        df = model.transform(df)
        df = df.select(['ctr', 'features']).withColumnRenamed('ctr','label')

        df.write.parquet(file)

    workingSet[df_o] = workingSet['ss'].read.parquet(file)
    
    logMessage(f'Finished Categorical Feature Encoding in {time()-start:.1f} Seconds')

### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def catPickFeatures(workingSet, subset = 'toy', threshold = 1800000, features = 300):

    start = time()

    file  = f'{workingSet["data"]}/criteo.parquet.{subset}.filled.masked-{threshold}.encode.picked-{features}'
    df_i  = f'df_{subset}_{threshold}_encode'
    df_o  = f'df_{subset}_{threshold}_picked'

    logMessage(f'Starting Feature Selection on {subset}')
    
    if  not exists(file):
        
        selector = ChiSqSelector(numTopFeatures = features, featuresCol = 'features', outputCol = 'pickedFeatures', labelCol = 'label')

        df       = workingSet[df_i]
        df       = selector.fit(df).transform(df)
        df       = df.select(['label', 'pickedFeatures']).withColumnRenamed('pickedFeatures','features')
        
        df.write.parquet(file)
      # df.write.partitionBy('label').parquet(file)

    workingSet[df_o] = workingSet['ss'].read.parquet(file)

    logMessage(f'Finished Feature Selection in {time()-start:.1f} Seconds')

### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def catHashFeatures(workingSet, subset = 'toy', threshold = 180000):

    start = time()

    file  = f'{workingSet["data"]}/criteo.parquet.{subset}.filled.masked-{threshold}.hashed'
    df_i  = f'df_{subset}_{threshold}_picked'
    df_o  = f'df_{subset}_{threshold}_hashed'

    features = 2 ** 9

    logMessage(f'Starting Categorical Feature Hashing on {subset} with {features} numFeatures')
    
    if  not exists(file):

        df     = workingSet[df_i]
        
        hasher = FeatureHasher(inputCols = workingSet['cat_features'], outputCol = 'features', numFeatures = features )

        df     = hasher.transform(df)
        df     = df.select(['ctr', 'features']).withColumnRenamed('ctr','label')

        df.write.parquet(file)
      # df.write.partitionBy('label').parquet(file)

    workingSet[df_o] = workingSet['ss'].read.parquet(file)

    logMessage(f'Finished Categorical Feature Hashing in {time()-start:.1f} Seconds')
