from code.engineering import *

Engineering.setupSpark(application = 'prep')
Engineering.importData(location = 'data', clean = False)
Engineering.splitsData(ratios = [.8, .1, .1])

min = 60000 # 1150
top =   987 #  987 + 13 = 1000

Engineering.numDoMeasurement(subset = 'train', iStep = f'')

Engineering.allDoStandardize(subset = 'train', iStep = f'')
Engineering.allDoStandardize(subset = 'tests', iStep = f'')
Engineering.allDoStandardize(subset = 'valid', iStep = f'')

Engineering.catDoMeasurement(subset = 'train', iStep = f'normed')

Engineering.catMaskUncommons(subset = 'train', iStep = f'normed', min = min)
Engineering.catMaskUncommons(subset = 'tests', iStep = f'normed', min = min)
Engineering.catMaskUncommons(subset = 'valid', iStep = f'normed', min = min)

Engineering.catDoCodeFeature(subset = 'train', iStep = f'normed.masked-{min:06d}', fit = True)
Engineering.catDoCodeFeature(subset = 'tests', iStep = f'normed.masked-{min:06d}')
Engineering.catDoCodeFeature(subset = 'valid', iStep = f'normed.masked-{min:06d}')

Engineering.allDoPackFeature(subset = 'train', iStep = f'normed.masked-{min:06d}.encode', fit = True)
Engineering.allDoPackFeature(subset = 'tests', iStep = f'normed.masked-{min:06d}.encode')
Engineering.allDoPackFeature(subset = 'valid', iStep = f'normed.masked-{min:06d}.encode')
