#%%
from torchvision.models import alexnet

lexnet_model = alexnet(pretrained=True)

print(lexnet_model)

#%%
import torch
imput = torch.randn(1, 3, 256, 256)
# %%
with torch.no_grad():
    output = lexnet_model(imput)
    print(output.shape)
# %%
