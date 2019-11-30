from common      import *
from engineering import *

initSpark(workingSet)
loadData( workingSet)
splitData(workingSet)

catFillNA(workingSet, subset = 'train', term = 'deadbeef') # replace undefined terms with deadbeef
catSwapIF(workingSet, subset = 'train', term = 'rarebeef') # replace infrequent terms with rarebeef
