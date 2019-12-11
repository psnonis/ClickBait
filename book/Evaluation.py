from code.evaluation import *

Evaluation.setupSpark(application = 'eval')
Evaluation.importData(location = 'data', clean = False)

Evaluation.eval(estimator = LogisticRegression,
                evaluator = BinaryClassificationEvaluator,
                dataframe = f'normed.filled.masked-060000.encode.packed',
                weighting = False, maxIter = 100, regParam = 0.0, elasticNetParam = 0, family = 'binomial')

Evaluation.eval(estimator = LogisticRegression,
                evaluator = BinaryClassificationEvaluator,
                dataframe = f'normed.filled.masked-060000.encode.packed',
                weighting = False, maxIter = 100, regParam = 0.0, elasticNetParam = 1, family = 'binomial')