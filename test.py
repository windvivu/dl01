#%%
from torchvision import datasets as dset
#%%
trainset = dset.CIFAR100(root='./data', train=True, download=True)


# %%
