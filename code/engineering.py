from code.common import *

### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def numCalculateStatistics(workingSet, subset, fit = False):
    """
    Calculate descriptive stats on training data for use in standardization
    Input  : Raw training dataframe
    Output : Training stats
    """    
    
    start = time()
    
    df_i  = f'df.{subset}'
    
    logMessage(f'Starting Numerical Feature Statistics Calculation on {subset}')
    
    if  fit:

        df             = workingSet[df_i]
        num_features   = workingSet['num_features']
      # run describe on the numerical features, then put transposed version of results into Pandas
        num_statistics = df.describe(num_features).toPandas().T
      # calculate median, adjust indices and column names and add median to pandas dataframe
        num_medians    = df.approxQuantile(col = num_features, probabilities = [0.5], relativeError = 0.005)
        num_medians    = list(chain.from_iterable(num_medians))
        num_statistics = num_statistics.rename(columns = num_statistics.iloc[0])
        num_statistics = num_statistics.drop(num_statistics.index[0])

        num_statistics['median'] = num_medians

        for s in num_statistics.columns:
            num_statistics[s] = pd.to_numeric(num_statistics[s], downcast = 'float')
       
        workingSet['num_statistics'] = num_statistics

    logMessage(f'Finished Numerical Feature Statistics Calculation in {time()-start:.1f} Seconds')

### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def numStandardizeFeatures(workingSet, subset):
    """
    Apply standardizations to all numerical features to ensure numerical features have balanced weights
    
    Input  : Spark Sql Dataframe of original labled data and relevant statistics from training data
    Output : Spark Sql Dataframe of all labeled data with standardized numeric features
    """

    start  = time()
    prefix = workingSet['prefix']

    df_i   = f'df.{subset}'
    df_o   = f'df.{subset}.normed'

    file   = f'{prefix}.{df_o}'
    
    logMessage(f'Starting Numerical Feature Standardization on {subset}')

    if  not exists(file):
        
        df = workingSet[df_i]
        
        num_statistics = workingSet['num_statistics'].to_dict()

        num_features   = workingSet['num_features'] #              numerical   features
        std_features   = workingSet['std_features'] # standardized numerical   features
        cat_features   = workingSet['cat_features'] #              categorical features

      # replace all undefined values with the median for that feature
        df = df.fillna(num_statistics['median'])

      # add standardized numerical feature columns  
        df = df.withColumn('i01_standard',((df['i01']                                )/(2.0*num_statistics['stddev']['i01'])))
        df = df.withColumn('i02_standard',((df['i02']-num_statistics['median']['i02'])/(1.0*num_statistics['stddev']['i02'])))
        df = df.withColumn('i03_standard',((df['i03']                                )/(1.0*num_statistics['stddev']['i03'])))
        df = df.withColumn('i04_standard',((df['i04']-num_statistics['median']['i04'])/(1.0*num_statistics['stddev']['i04'])))
        df = df.withColumn('i05_standard',((df['i05']-num_statistics['median']['i05'])/(1.0*num_statistics['stddev']['i05'])))
        df = df.withColumn('i06_standard',((df['i06']                                )/(2.0*num_statistics['stddev']['i06'])))
        df = df.withColumn('i07_standard',((df['i07']                                )/(2.0*num_statistics['stddev']['i07'])))
        df = df.withColumn('i08_standard',((df['i08']                                )/(2.0*num_statistics['stddev']['i08'])))
        df = df.withColumn('i09_standard',((df['i09']-num_statistics['median']['i09'])/(1.0*num_statistics['stddev']['i09'])))
        df = df.withColumn('i10_standard',((df['i10']                                )/(1.0*num_statistics['max'   ]['i10'])))
        df = df.withColumn('i11_standard',((df['i11']-num_statistics['median']['i11'])/(1.0*num_statistics['stddev']['i11'])))
        df = df.withColumn('i12_standard',((df['i12']                                )/(2.0*num_statistics['stddev']['i12'])))
        df = df.withColumn('i13_standard',((df['i13']-num_statistics['median']['i13'])/(1.0*num_statistics['stddev']['i13'])))

        assembler_num = VectorAssembler(inputCols = num_features, outputCol = 'num_features')
        assembler_std = VectorAssembler(inputCols = std_features, outputCol = 'std_features')

        pipeline      = Pipeline(stages = [assembler_num, assembler_std])

        df            = pipeline.fit(df).transform(df)
        df            = df.select('label', 'num_features', 'std_features', *cat_features)

        df.write.parquet(file)
        
    workingSet[df_o] = workingSet['ss'].read.parquet(file)

    logMessage(f'Finished Numerical Feature Standardization in {time()-start:.1f} Seconds')

### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def catFillUndefined(workingSet, subset = 'toy', term = 'deadbeef'):

    start  = time()
    prefix = workingSet['prefix']

    df_i   = f'df.{subset}.normed'
    df_o   = f'df.{subset}.normed.filled'

    file   = f'{prefix}.{df_o}'

    logMessage(f'Starting Categorical Fill Undefined Terms on {subset}')

    if  not exists(file):
        
        df = workingSet[df_i]

      # df = df.drop(*workingSet['num_features'])

        df = df.fillna(term, workingSet['cat_features'])

        df.write.parquet(file)

    workingSet[df_o] = workingSet['ss'].read.parquet(file)

    logMessage(f'Finished Categorical Fill Undefined Terms in {time()-start:.1f} Seconds')

### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def catFindFrequent(workingSet, subset = 'toy', threshold = 180000, fit = False):
    
    start  = time()
    prefix = workingSet['prefix']

    df_i   = f'df.{subset}.normed.filled'
    df_e   = f'df.{subset}.normed.filled.masked-{threshold}'
    
    logMessage(f'Starting Categorical Find Frequent Terms on {subset}')
    
    if  fit:
        df = workingSet[df_i]
    else:
        df = workingSet[df_e] # evaluvate masked dataframe

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

    start  = time()
    prefix = workingSet['prefix']

    df_i   = f'df.{subset}.normed.filled'
    df_o   = f'df.{subset}.normed.filled.masked-{threshold}'

    file   = f'{prefix}.{df_o}'
    
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

    start  = time()
    prefix = workingSet['prefix']

    df_i   = f'df.{subset}.normed.filled.masked-{threshold}'
    df_o   = f'df.{subset}.normed.filled.masked-{threshold}.encode'

    file   = f'{prefix}.{df_o}'

    logMessage(f'Starting Categorical Feature Encoding on {subset}')
    
    df = workingSet[df_i]

    if  fit:

        stages   = []

        features = workingSet['cat_features']
        indexes  = [f'{f}_index'  for f in features]
        vectors  = [f'{f}_vector' for f in features]

        for feature, index, vector in zip(features, indexes, vectors):
            indexer  = StringIndexer(inputCol = feature, outputCol = index)
            encoder  = OneHotEncoderEstimator(inputCols = [indexer.getOutputCol()], outputCols = [vector], dropLast = False) # handleInvalid = 'keep'
            stages  += [indexer, encoder]

        assembler = VectorAssembler(inputCols = vectors, outputCol = 'cat_features')
        stages   += [assembler]

        pipeline  = Pipeline(stages = stages)
        model     = pipeline.fit(df)
                       
        workingSet['code_model'] = model
                       
    else:

        model = workingSet['code_model']
    
    if  not exists(file):
        
        df = model.transform(df)

        df = df.select(['label', 'num_features', 'std_features', 'cat_features'])

        df.write.parquet(file)

    workingSet[df_o] = workingSet['ss'].read.parquet(file)
    
    logMessage(f'Finished Categorical Feature Encoding in {time()-start:.1f} Seconds')

### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def catPickFeatures(workingSet, subset = 'toy', threshold = 1800000, top = 300):

    start  = time()
    prefix = workingSet['prefix']

    df_i   = f'df.{subset}.normed.filled.masked-{threshold}.encode'
    df_o   = f'df.{subset}.normed.filled.masked-{threshold}.encode.picked-{top}'

    file   = f'{prefix}.{df_o}'

    logMessage(f'Starting Feature Selection on {subset} for {top} features')
    
    if  not exists(file):
        
        selector = ChiSqSelector(numTopFeatures = top, featuresCol = 'cat_features', outputCol = 'top_features', labelCol = 'label')

        df       = workingSet[df_i]
        df       = selector.fit(df).transform(df)

        df.write.parquet(file)
      # df.write.partitionBy('label').parquet(file)

    workingSet[df_o] = workingSet['ss'].read.parquet(file)

    logMessage(f'Finished Feature Selection in {time()-start:.1f} Seconds')

### ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def catHashFeatures(workingSet, subset = 'toy', threshold = 180000, top = 300):

    start  = time()
    prefix = workingSet['prefix']

    df_i   = f'df.{subset}.normed.filled.masked-{threshold}.picked-{top}'
    df_o   = f'df.{subset}.normed.filled.masked-{threshold}.picked-{top}.hashed'

    file   = f'{prefix}.{df_o}'
    
    width  = 2 ** 9

    logMessage(f'Starting Categorical Feature Hashing on {subset} with {features} numFeatures')
    
    if  not exists(file):

        df     = workingSet[df_i]
        
        hasher = FeatureHasher(inputCols = workingSet['cat_features'], outputCol = 'features', numFeatures = width)

        df     = hasher.transform(df)
        df     = df.select(['label', 'features'])

        df.write.parquet(file)
      # df.write.partitionBy('label').parquet(file)

    workingSet[df_o] = workingSet['ss'].read.parquet(file)

    logMessage(f'Finished Categorical Feature Hashing in {time()-start:.1f} Seconds')

def allPackFeatures(workingSet, subset = 'toy', threshold = 180000, top = 300):
    
    start  = time()
    prefix = workingSet['prefix']

    df_i   = f'df.{subset}.normed.filled.masked-{threshold}.encode.picked-{top}'
    df_o   = f'df.{subset}.normed.filled.masked-{threshold}.encode.picked-{top}.packed'

    file   = f'{prefix}.{df_o}'

    logMessage(f'Starting Feature Packing on {subset}')
    
    if  not exists(file):

        df        = workingSet[df_i]

        assembler = VectorAssembler(inputCols = ['std_features', 'top_features'], outputCol = ['features'])
        
        df        = assembler.transform(df)
        
        df        = df.select('label', 'features')

        df.write.parquet(file)
        
    workingSet[df_o] = workingSet['ss'].read.parquet(file)
    
    logMessage(f'Finished Feature Packing in {time()-start:.1f} Seconds')
