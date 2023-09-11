#%%
from torchvision.models import alexnet
from torchvision.models import AlexNet_Weights
import simplecnn

lexnet_model = alexnet(weights=AlexNet_Weights.DEFAULT)

print(lexnet_model)

# %%

print(lexnet_model.classifier)

# %%
lexnet_model.classifier[0]
# %%
import torch.nn as nn

a = nn.Sequential(lexnet_model.classifier[0],
                  lexnet_model.classifier[1],
                  lexnet_model.classifier[2],
                  lexnet_model.classifier[3],
                  
                  )
# %%
lexnet_model.classifier = a
# %%
print(lexnet_model)
# %%
