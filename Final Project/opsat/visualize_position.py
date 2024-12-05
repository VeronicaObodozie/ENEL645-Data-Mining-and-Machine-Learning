from matplotlib import pyplot as plt
from random import randint
from utils import *
%matplotlib inline

dataset_root_dir = 'datasets/' # path to speed+'
dataset = SatellitePoseEstimationDataset(root_dir=dataset_root_dir)

rows = 4
cols = 2

fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
for i in range(rows):
    for j in range(cols):
        dataset.visualize(randint(0, 12000), ax=axes[i][j])
        axes[i][j].axis('off')
fig.tight_layout()