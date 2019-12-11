from code.engineering import *

Engineering.setupSpark(application = 'prep')
Engineering.importData(location = 'data', clean = False)
Engineering.splitsData(ratios = [.8, .1, .1])

min = 60000 # 1150
top =   987 #  987 + 13 = 1000

Engineering.numDoMeasurement(subset = 'train', iStep = f'', fit = True)

Engineering.numDoStandardize(subset = 'train', iStep = f'')
Engineering.numDoStandardize(subset = 'tests', iStep = f'')
Engineering.numDoStandardize(subset = 'valid', iStep = f'')

Engineering.catFillUndefined(subset = 'train', iStep = f'normed')
Engineering.catFillUndefined(subset = 'tests', iStep = f'normed')
Engineering.catFillUndefined(subset = 'valid', iStep = f'normed')

Engineering.catFindFrequents(subset = 'train', iStep = f'normed.filled', min = min, fit = True)

Engineering.catMaskUncommons(subset = 'train', iStep = f'normed.filled', min = min)
Engineering.catMaskUncommons(subset = 'tests', iStep = f'normed.filled', min = min)
Engineering.catMaskUncommons(subset = 'valid', iStep = f'normed.filled', min = min)

Engineering.catDoCodeFeature(subset = 'train', iStep = f'normed.filled.masked-{min:06d}', fit = True)
Engineering.catDoCodeFeature(subset = 'tests', iStep = f'normed.filled.masked-{min:06d}')
Engineering.catDoCodeFeature(subset = 'valid', iStep = f'normed.filled.masked-{min:06d}')

Engineering.allDoPackFeature(subset = 'train', iStep = f'normed.filled.masked-{min:06d}.encode', fit = True)
Engineering.allDoPackFeature(subset = 'tests', iStep = f'normed.filled.masked-{min:06d}.encode')
Engineering.allDoPackFeature(subset = 'valid', iStep = f'normed.filled.masked-{min:06d}.encode')
