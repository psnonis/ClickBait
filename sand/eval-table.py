from code.evaluation import *

Evaluation.setupSpark(application = 'eval')

frame = 'normed.masked-060000.encode.picked-000987.action.packed-016463'
frame =        'normed.masked-060000.encode.picked-000987.packed-001000'
frame =                      'normed.masked-001000.encode.packed-036168'
frame =                      'normed.masked-000500.encode.packed-056507'
frame =               'normed.masked-001000.encode.action.packed-506183'

frame =        'normed.masked-001000.encode.action.packed-506183'

Evaluation.eval(estimator = LogisticRegression,
                evaluator = BinaryClassificationEvaluator,
                dataframe = frame,
                weighting = False, maxIter = 100, regParam = 0.0, elasticNetParam = 0, family = 'binomial')
