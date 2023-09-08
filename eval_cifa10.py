import torch
import torch.nn as nn
import os
from PIL import Image
import torchvision.transforms as transforms

img_path = "img/8.jpeg"

model_path = "model/cifa10_bestcheckpoint.pt"
cat = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
if os.path.exists(model_path):
    print("Loading model...")
else:
    print("Model not exists!")
    exit()

try:
    checkpoint = torch.load(model_path)
except:
    print("Model load failed!")
    exit()

print("Model load success!")
print("Model info:")
print(" Epoch: ", checkpoint["epoch"])
print(" Accuracy: ", checkpoint["accu"])
print("--------------------------------")
model = checkpoint["model"]
model.to("cpu")
model.eval()
# %%
img = Image.open(img_path).convert("RGB")
input = img.resize((32,32))
input = transforms.ToTensor()(input)
input = input.unsqueeze(0)
with torch.no_grad():
    output = model(input)

confident = nn.Softmax(dim=1)(output)
confident = format(torch.max(confident).item(),'.2%')
maxindex = torch.argmax(output).item()
print('Predicted:', cat[maxindex].upper(), 'with probability:', confident)


