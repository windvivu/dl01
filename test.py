#%%
import numpy as np
import torch
# %%
v1=np.array([1,2,3])
v2=np.array([4,5,6])

kq=np.dot(v1,v2)

# %%
tv1=torch.tensor(v1, dtype=torch.float32)
tv2=torch.tensor(v2, dtype=torch.float32)
# %%
tkq=torch.dot(tv1,tv2)
# %%
v2d1 = torch.tensor([[1,2,3],[4,5,6]])
v2d2 = torch.tensor([[1,2,1],[3,4,1]])
# %%
v2d3 = torch.tensor([[1,2],[3,4]])
# %%
