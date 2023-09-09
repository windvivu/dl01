#%%
from torchvision.models import alexnet

lexnet_model = alexnet(weights=True)

print(lexnet_model)

#%%
import torch
imput = torch.randn(1, 3, 128, 128)
# %%
with torch.no_grad():
    output = lexnet_model(imput)
    print(output.shape)
# %%
