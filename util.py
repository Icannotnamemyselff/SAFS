import numpy as np
from collections import OrderedDict


def mask_generator (p_m, x):
    """Generate mask vector.

    Args:
      - p_m: corruption probability
      - x: feature matrix

    Returns:
      - mask: binary mask matrix
    """
    mask = np.random.binomial(1, p_m, x.shape)
    return mask

def initLabeled(y,seed,args):
    ## random selected the labeled instances' index
    y = np.squeeze(y)
    np.random.seed(seed)
    labeledIndex = []
    labelDict = OrderedDict()
    for label in np.unique(y):
        labelDict[label] = []
    for i, label in enumerate(y):
        labelDict[label].append(i)
    for value in labelDict.values():
        for idx in np.random.choice(value, size=args.select_sample, replace=False, p=None):
            labeledIndex.append(idx)
    # print(y[labeledIndex])
    return labeledIndex

def initLabeled_ini(y,seed,size):
    ## random selected 5000
    y = np.squeeze(y)
    np.random.seed(seed)
    labeledIndex = []
    labelDict = OrderedDict()
    for label in np.unique(y):
        # if label <5:
            labelDict[label] = []
    for i, label in enumerate(y):
        # if label <5:
            labelDict[label].append(i)
    for value in labelDict.values():
        for idx in np.random.choice(value, size=size, replace=False, p=None):
            labeledIndex.append(idx)
    # print(y[labeledIndex])
    return labeledIndex

def initLabeled_few_shot(y,seed,args):
    ## random selected the labeled instances' index
    y = np.squeeze(y)
    np.random.seed(seed)
    labeledIndex = []
    labelDict = OrderedDict()
    for label in np.unique(y):
        # if label <5:
            labelDict[label] = []
    for i, label in enumerate(y):
        # if label <5:
            labelDict[label].append(i)
    for value in labelDict.values():
        for idx in np.random.choice(value, size=args.k_shot, replace=False, p=None):
            labeledIndex.append(idx)
    return labeledIndex