from code.common import *

iFile: str = None
oStep: str = None
oFile: str = None
oPipe: str = None

class Engineering(Common):

    def stepStarting(step: str, message: str, subset: str, iStep: str = '', fit:bool = False, model: str = '') -> DataFrame:

        global iFile, oFile, oStep, oPipe

        iFile = f'{Common.prefix}/{subset}.parquet.{iStep}'.strip('.')
        oStep = f'{                          iStep}.{step}'.strip('.')
        oFile = f'{Common.prefix}/{subset}.parquet.{oStep}'.strip('.')
        oPipe = f'{Common.prefix}/model.pickled.{   model}'.strip('.')
        
        if  not Common.spark:

            Common.sparkSetup(application = 'engineering')
            
        if  exists(iFile) and not exists(f'{iFile}/_SUCCESS'):
            system(f'rm -rf {iFile}')

        if  exists(oFile) and not exists(f'{oFile}/_SUCCESS'):
            system(f'rm -rf {oFile}')

        if  exists(iFile) :
            
            if  not exists(oFile) :
                
                if  Common.spark:

                    timePrint(f'Starting {message} : iFile = {iFile}')

                    return Common.spark.read.parquet(iFile)

                else:

                    timePrint(f'Skipping {message} : iFile = {iFile} : Spark Not Ready')

                    return None

            elif fit and not exists(oPipe):

                    timePrint(f'Starting {message} : iFile = {iFile}')

                    return Common.spark.read.parquet(iFile)
                
            else:

                timePrint(f'Skipping {message} : iFile = {iFile}')

                return None

        else:

            timePrint(f'Skipping {message} : iFile = {iFile} : iFile Not Found')

            return None

    def stepStopping(step: str, message: str, subset: str, oData: DataFrame = None):

        global iFile, oFile, oStep, oPipe

        if  oData != None:

            oData.write.parquet(oFile)

        timePrint(f'Stopping {message} : oFile = {oFile}\n')
    
    def numDoMeasurement(subset: str, iStep: str):
        """
        Calculate descriptive stats on training data for use in standardization
        Input  : Raw training dataframe
        Output : Numerical training stats
        """    

        global iFile, oFile, oStep, oPipe

        oData = None
        iData = Engineering.stepStarting('', 'Numerical Data Measurement', subset, iStep, fit = True)
        oPipe = f'{Common.prefix}/model.pickled.num_measures'

        if  iData and not exists(oPipe):

          # run describe on the numerical features, then put transposed version of results into Pandas
            num_measures = iData.describe(Common.num_features).toPandas().T
          # calculate median, adjust indices and column names and add median to pandas dataframe
            num_medians  = iData.approxQuantile(col = Common.num_features, probabilities = [0.5], relativeError = 0.005)
            num_medians  = list(chain.from_iterable(num_medians))
            num_measures = num_measures.rename(columns = num_measures.iloc[0])
            num_measures = num_measures.drop(num_measures.index[0])

            num_measures['median'] = num_medians

            for c in num_measures.columns:
                num_measures[c] = pd.to_numeric(num_measures[c], downcast = 'float')

            dump(num_measures,     open(oPipe, 'wb'))

        Common.num_measures = load(open(oPipe, 'rb'))

        Engineering.stepStopping('', 'Numerical Data Measurement', subset, oData)
        
    def allDoStandardize(subset: str, iStep: str):
        """
        Apply standardizations to all numerical features to ensure numerical features have balanced weights.
        Replace all undefined categorical values with deadbeef.

        Input  : Spark Sql Dataframe of original labled data and relevant statistics from training data
        Output : Spark Sql Dataframe of all labeled data with standardized numeric features and filled categoric features
        """

        global iFile, oFile, oStep

        oData = None
        iData = Engineering.stepStarting('normed', 'Numerical Data Standardize', subset, iStep)

        if  iData != None:

            num_measures = Common.num_measures.to_dict()

          # replace all undefined values with the median for that feature
            oData = iData.fillna(num_measures['median'])
            
          # add standardized numerical feature columns  
            oData = oData.withColumn('n01_standard',((oData['n01']                              )/(2.0*num_measures['stddev']['n01'])))
            oData = oData.withColumn('n02_standard',((oData['n02']-num_measures['median']['n02'])/(1.0*num_measures['stddev']['n02'])))
            oData = oData.withColumn('n03_standard',((oData['n03']                              )/(1.0*num_measures['stddev']['n03'])))
            oData = oData.withColumn('n04_standard',((oData['n04']-num_measures['median']['n04'])/(1.0*num_measures['stddev']['n04'])))
            oData = oData.withColumn('n05_standard',((oData['n05']-num_measures['median']['n05'])/(1.0*num_measures['stddev']['n05'])))
            oData = oData.withColumn('n06_standard',((oData['n06']                              )/(2.0*num_measures['stddev']['n06'])))
            oData = oData.withColumn('n07_standard',((oData['n07']                              )/(2.0*num_measures['stddev']['n07'])))
            oData = oData.withColumn('n08_standard',((oData['n08']                              )/(2.0*num_measures['stddev']['n08'])))
            oData = oData.withColumn('n09_standard',((oData['n09']-num_measures['median']['n09'])/(1.0*num_measures['stddev']['n09'])))
            oData = oData.withColumn('n10_standard',((oData['n10']                              )/(1.0*num_measures['max'   ]['n10'])))
            oData = oData.withColumn('n11_standard',((oData['n11']-num_measures['median']['n11'])/(1.0*num_measures['stddev']['n11'])))
            oData = oData.withColumn('n12_standard',((oData['n12']                              )/(2.0*num_measures['stddev']['n12'])))
            oData = oData.withColumn('n13_standard',((oData['n13']-num_measures['median']['n13'])/(1.0*num_measures['stddev']['n13'])))

            assembler_num = VectorAssembler(inputCols = Common.num_features, outputCol = 'num_features')
            assembler_std = VectorAssembler(inputCols = Common.std_features, outputCol = 'std_features')

            pipeline      = Pipeline(stages = [assembler_num, assembler_std])
            model         = pipeline.fit(oData)

            oData         = model.transform(oData)
            oData         = oData.select('label', 'std_features', *Common.cat_features).fillna('deadbeef', Common.cat_features)

        Engineering.stepStopping('normed', 'Numerical Data Standardize', subset, oData)

    def catDoMeasurement(subset: str, iStep: str):
        """
        Calculate descriptive stats on training data for use in standardization
        Input  : Raw training dataframe
        Output : Categorical training stats
        """    

        global iFile, oFile, oStep, oPipe

        oData = None
        iData = Engineering.stepStarting('', 'Categorical Data Measurement', subset, iStep, fit = True)
        oPipe = f'{Common.prefix}/model.pickled.cat_measures'

        if  iData and not exists(oPipe):

            cat_measures = {}

            for feature in Common.cat_features:
                cat_measures[feature] = iData.select(feature).groupBy(feature).count() \
                    .sort('count', ascending = False) \
                    .withColumnRenamed(feature,  'value') \
                    .withColumnRenamed('count', 'counts') \
                .toPandas()

            dump(cat_measures,     open(oPipe, 'wb'))

        Common.cat_measures = load(open(oPipe, 'rb'))

        Engineering.stepStopping('', 'Categorical Data Measurement', subset, oData)
        
    def catMaskUncommons(subset: str, iStep: str, min: int = 100000):

        global iFile, oFile, oStep, oPipe

        oData = None
        iData = Engineering.stepStarting(f'masked-{min:06d}', f'Categorical Mask Uncommons with Threshold of <  {min}', subset, iStep)

        if  iData != None:

            oData  = iData

            """
            for feature, uncommon_categories in Common.cat_measures['uncommon'].items():
                oData = oData.replace(uncommon_categories, 'rarebeef')
            """

            total = 0
            
            for feature, measure in Common.cat_measures.items():
                
                frequent = measure[measure.counts > min].value.to_list()
                oData    = oData.withColumn(feature, when(~oData[feature].isin(*frequent), 'rarebeef').otherwise(oData[feature]))
                count    = oData.select(feature).groupBy(feature).count().count()
                total   += count
                
              # timePrint(f'{feature} : {count:>6} : {total:>6}')
                
        Engineering.stepStopping(f'masked-{min:06d}', f'Categorical Mask Uncommons with Threshold of <  {min}', subset, oData)

    def catDoCodeFeature(subset: str, iStep: str, fit: bool = False):

        global iFile, oFile, oStep, oPipe

        oData = None
        iData = Engineering.stepStarting('encode', 'Categorical One-Hot Encoding', subset, iStep, fit, 'encodingPipe')

        if  fit and not exists(oPipe) and iData != None:
           
            stages   = []
            indexes  = [f'{f}_idx' for f in Common.cat_features]
            vectors  = [f'{f}_vec' for f in Common.cat_features]

            for feature, index, vector in zip(Common.cat_features, indexes, vectors):
                indexer  = StringIndexer(inputCol = feature, outputCol = index)
                encoder  = OneHotEncoderEstimator(inputCols = [indexer.getOutputCol()], outputCols = [vector], dropLast = False)
                stages  += [indexer, encoder]

            assembler = VectorAssembler(inputCols = vectors, outputCol = 'cat_features')
            stages   += [assembler]

            encoding_pipe  = Pipeline(stages = stages)
            encoding_model = encoding_pipe.fit(iData)

            encoding_model.save(oPipe)

        Common.encode_model = PipelineModel.load(oPipe)

        if  iData != None and not exists(oFile):
            
            oData  = Common.encode_model.transform(iData)
            oData  = oData.select(['label', 'std_features', 'cat_features'])

        Engineering.stepStopping('encode', 'Categorical One-Hot Encoding', subset, oData)

    def catDoPickFeature(subset: str, iStep: str, fit: bool = False, top: int = 1000):

        global iFile, oFile, oStep, oPipe

        oData = None
        iData = Engineering.stepStarting(f'picked-{top:06d}', f'Categorical Selection for Top {top} Features', subset, iStep, fit, f'selectingPipe-{top:06d}')

        width = iData.select('cat_features').first().cat_features.size

        timePrint(f'Reducing from {width} Features to {top} Features')

        if  fit and not exists(oPipe) and iData != None and top:

            selector       = ChiSqSelector(numTopFeatures = top, featuresCol = 'cat_features', outputCol = 'top_features', labelCol = 'label')
            selector_pipe  = Pipeline(stages = [selector])
            selector_model = selector_pipe.fit(iData)

            selector_model.save(oPipe)

        Common.select_model = PipelineModel.load(oPipe)

        if  iData != None and top:

            oData  = Common.select_model.transform(iData)

        Engineering.stepStopping(f'picked-{top:06d}', f'Categorical Selection for Top {top} Features', subset, oData)

    def allDoPackFeature(subset: str, iStep: str, fit: bool = False):
        
        global iFile, oFile, oStep, oPipe

        oData = None
        iData = Engineering.stepStarting('packed', 'Packed Final Features', subset, iStep, fit, 'balance_rate')

        if  fit and not exists(oPipe) and iData != None:
            
            total        = iData.count()
            positive     = iData.filter('label == 1.0').count()
            balance_rate = positive / total

            dump(balance_rate,     open(oPipe, 'wb'))

        Common.balance_rate = load(open(oPipe, 'rb'))

        if  iData != None:

            features  = ['std_features']
            features += ['top_features'] if 'top_features' in iData.columns else \
                        ['cat_features']
            features += ['cxn_features'] if 'cxn_features' in iData.columns else \
                        []

            assembler    = VectorAssembler(inputCols = features, outputCol = 'features')
            
            oData        = assembler.transform(iData)
            oData        = oData.select('label', 'features')
            oData        = oData.withColumn('weight', when(oData.label == 0.0, 1.0 * Common.balance_rate).otherwise(1.0 - Common.balance_rate))
            
        Engineering.stepStopping('packed', 'Packed Final Features', subset, oData)
        
    def toyTakeSubSample(subset: str, iStep: str, len: int = 1000):
        
        global iFile, oFile, oStep, oPipe

        oData = None
        iData = Engineering.stepStarting('toy', 'Take Toy Sample', subset, iStep)
        
        if  iData != None:
            
            oData  = iData.orderBy(rand(seed = 2019)).limit(len)
            
            width  = oData.head().features.size
            count  = oData.count()
            click  = oData.filter("label==1").count()
            ratio  = f'{click/count*100:.2f}'
            
            timePrint(f'toy {subset} {width}x{count} = {ratio}% clicks')

        Engineering.stepStopping('toy', 'Take Toy Sample', subset, oData)
