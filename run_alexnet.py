#%%
from torchvision import datasets as dset
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from torchvision.models import alexnet
from torch import nn

namedata = "alexnet_cifa100"
num_class = 100

transform = Compose([Resize((256, 256)), ToTensor(),Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
trainset = dset.CIFAR100(root='data', train=True, download=True, transform=transform)
testset = dset.CIFAR100(root='data', train=False, download=True, transform=transform)

#%%
batch_size = 32
num_epochs = 10000

#%%
import os
import shutil
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
from tqdm import tqdm
from simplecnn import SimpleCNN
# from lenet import LeNet
import torch

# save model
def savecheckpoint(model, filename, epoch, accu, optimizer=None):
    checkpoint = {
      "model": model,
      "epoch": epoch,
      "accu": accu,
      "optimizer": None if optimizer == None else optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
#%%

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

savepath = "model"
logpath = "trainlogging"
if not os.path.exists(savepath):
    os.makedirs(savepath)
if os.path.exists(logpath):
    shutil.rmtree(logpath)

if os.path.exists(os.path.join(savepath, namedata + "_bestcheckpoint.pt")) and os.path.exists(os.path.join(savepath, namedata + "_bestcheckpoint_accu.txt")):
    with open(os.path.join(savepath, namedata + "_bestcheckpoint_accu.txt"), "r") as f:
        try:
            bestaccu = float(f.readline())
        except:
            bestaccu = 0
else:
    bestaccu = 0

writer = SummaryWriter(logpath)
# %%

train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

total_batch = len(train_dataloader)
total_batch_test = len(test_dataloader)
# %%
if os.path.exists(os.path.join(savepath, namedata + "_lastcheckpoint.pt")):
    checkpoint = torch.load(os.path.join(savepath, namedata + "_lastcheckpoint.pt"))
    model = checkpoint["model"]
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]
    print("Load last checkpoint success!")
    if start_epoch >= num_epochs:
        print("Training completed!")
        exit()
else:
    start_epoch = 1
    model = alexnet(pretrained=True).to(device)
    model.classifier[6] = nn.Linear(4096, num_class).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

start_epoch -=1
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(start_epoch, num_epochs):
        print("Epoch: ", str(epoch+1), "/", str(num_epochs))
        model.train()
        progress_bar = tqdm(train_dataloader, desc=" Training", colour="green")
        for iter, (images, labels_train) in enumerate(progress_bar):
            images = images.to(device)
            labels_train = labels_train.to(device)

            outputs_train = model(images) # forward
            loss = criterion(outputs_train, labels_train) # loss

            writer.add_scalar("Loss/train", loss, epoch*total_batch+iter)

            optimizer.zero_grad() # gradient -> làm sach buffer gradient
            loss.backward() # backward
            optimizer.step() # update weight
        print(' End epoch', epoch+1,'loss value of training:', loss.item())

        model.eval()
        all_predictions = []
        all_labels = []
        for images, labels_test in tqdm(test_dataloader, desc=" Testing"):
            images = images.to(device)
            labels_test = labels_test.to(device)

            all_labels.extend(labels_test)
            with torch.no_grad():
                outputs_test = model(images) # forward
                max_to_label = torch.argmax(outputs_test, dim=1)
                #loss = criterion(outputs_test, labels_test) # loss
                all_predictions.extend(max_to_label)
        print(' End epoch', epoch+1, end=' - ')
        accu = (torch.tensor(all_predictions) == torch.tensor(all_labels)).sum().item()/len(all_labels)
        print('Accuracy:', accu)
        writer.add_scalar("Accuracy/test", accu, epoch+1)
        if accu > bestaccu:
            bestaccu = accu
            savecheckpoint(model, os.path.join(savepath, namedata + "_bestcheckpoint.pt"),epoch+1,accu)
            with open(os.path.join(savepath, namedata + "_bestcheckpoint_accu.txt"), "w") as f:
                f.write(str(bestaccu))
        savecheckpoint(model, os.path.join(savepath, namedata + "_lastcheckpoint.pt"),epoch+1,accu,optimizer)
        with open(os.path.join(savepath, namedata + "_lastcheckpoint_accu.txt"), "w") as f:
            f.write(str(accu))

all_labels_ = [i.item() for i in all_labels]
all_predictions_ = [i.item() for i in all_predictions]
print(classification_report(all_labels_, all_predictions_))
# %%
