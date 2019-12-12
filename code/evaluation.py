from code.common import *

from pyspark.ml.classification import LogisticRegression, LinearSVC
from pyspark.ml.evaluation     import BinaryClassificationEvaluator

from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation     import MulticlassClassificationEvaluator

class Evaluation(Common):

    def eval(estimator, evaluator, dataframe = 'normed.filled.masked-180000.encode.picked-000300.packed', features = 'features', weighting = False, **params):
        
        """
        Evaluate model performance on Train, Test, and Validation subsets.
        
        Input  : Estimator, Evaluator, Engineered Dataset.
        Output : AUC on Respective Subset.
        """

        timePrint('')
        timePrint('=' * 120)
        timePrint('')

        frames = {}
        widths = {}
        predis = {}
        scores = {}
        
        name   = estimator.__name__

        timePrint(f"{name} : Frames : {dataframe}")        
        timePrint(f"{name} : Weight : {weighting}")
        timePrint(f"{name} : Params : {params}")

        for subset in ['train', 'tests', 'valid']:

            frame = f'data/{subset}.parquet.{dataframe}'

            if  exists(frame):
                frames[subset] = Common.spark.read.parquet(frame).select('label', features)
                widths[subset] = frames[subset].select('features').first().features.size

        if  weighting:
           
            estimator = estimator(featuresCol = features, labelCol = 'label', weightCol = 'weight', **params)

        else:

            estimator = estimator(featuresCol = features, labelCol = 'label',                       **params)

        evaluator = evaluator()

        start     = time()

        if  frames.get('train'):
        
            timePrint(f"{name} :  Width : {widths['train']} Features")

            model = estimator.fit(frames['train'])

        training  = time()

        for subset in frames:
            
            predis[subset] = model.transform(frames[subset])

        predicts  = time()

        for subset in predis:
            
            scores[subset] = evaluator.evaluate(predis[subset])
            
        timePrint(f"{name} :  Elaps : {time() - start:.1f} Seconds")
        
        for subset in scores:
            
            timePrint(f"{name} :  {subset.title()} : Score = {scores[subset]*100:.2f}%")
