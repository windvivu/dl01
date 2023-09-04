#%%
from animaldata import Animal10data
from torch.utils.data import DataLoader

train_data = Animal10data(root='data/animalsv2', train=True)
test_data = Animal10data(root='data/animalsv2', train=False)

#%%
print(train_data.getitem_path(0))
# %%
