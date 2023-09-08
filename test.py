#%%
from torchvision import datasets as dset
#%%
trainset = dset.CIFAR10(root='./data', train=True, download=True)
print(trainset.classes)

# %%
# ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']