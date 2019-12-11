from code.evaluation import *

Evaluation.setupSpark(application = 'eval')
Evaluation.importData(   location = 'data', clean = False)

frame = 'normed.masked-060000.encode.picked-000987.packed-001000'

Evaluation.eval(estimator = LogisticRegression,
                evaluator = BinaryClassificationEvaluator,
                dataframe = frame,
                weighting = False, maxIter = 100, regParam = 0.0, elasticNetParam = 0, family = 'binomial')

Evaluation.eval(estimator = LogisticRegression,
                evaluator = BinaryClassificationEvaluator,
                dataframe = frame,
                weighting = False, maxIter = 100, regParam = 0.1, elasticNetParam = 0, family = 'binomial')

Evaluation.eval(estimator = LogisticRegression,
                evaluator = BinaryClassificationEvaluator,
                dataframe = frame,
                weighting = False, maxIter = 100, regParam = 0.5, elasticNetParam = 1, family = 'multinomial')
