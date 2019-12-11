from code.engineering import *

setupSpark(workingSet, application = 'prep')
importData(workingSet, location = 'data')
splitFrame(workingSet, ratios = [.8,.1,.1])

numCalculateStatistics(workingSet, subset = 'train', fit = True)

numStandardizeFeatures(workingSet, subset = 'train')
numStandardizeFeatures(workingSet, subset = 'dev'  )

catFillUndefined(workingSet, subset = 'train')
catFillUndefined(workingSet, subset = 'dev'  )

catFindFrequent(workingSet, subset = 'train', threshold = 180000, fit = True)

catMaskUncommon(workingSet, subset = 'train', threshold = 180000)
catMaskUncommon(workingSet, subset = 'dev',   threshold = 180000)

catCodeFeatures(workingSet, subset = 'train', threshold = 180000, fit = True)
catCodeFeatures(workingSet, subset = 'dev',   threshold = 180000)


catPickFeatures(workingSet, subset = 'train', threshold = 1800000, top = 400, fit = True)
catPickFeatures(workingSet, subset = 'dev',   threshold = 1800000, top = 400)

allPackFeatures(workingSet, subset = 'train', threshold = 180000, top = 400)
allPackFeatures(workingSet, subset = 'dev',   threshold = 180000, top = 400)