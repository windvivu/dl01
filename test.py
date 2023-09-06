#%%
from torchvision import datasets, transforms
#%%
trainset = datasets.CelebA(root='./data', split='train', download=True)
# %%
a,b = trainset[0]
# %%
from animaldata import Animal10data
# %%
trainset = Animal10data(root='./data/animalsv2', train=True)

# %%
