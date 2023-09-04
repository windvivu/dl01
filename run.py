#%%
import os
import shutil
from animaldata import Animal10data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
from tqdm import tqdm
from simplecnn import SimpleCNN
import torch


trainset = Animal10data(root='data/animalsv2', train=True)
testset = Animal10data(root='data/animalsv2', train=False)
bestaccu = 0

# save model
def savecheckpoint(model, filename):
    checkpoint = {
      "model": model,
      "categories": testset.categories
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

writer = SummaryWriter(logpath)
# %%

batch_size = 32
num_epochs = 100

train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

total_batch = len(train_dataloader)
total_batch_test = len(test_dataloader)
# %%
model = SimpleCNN().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(num_epochs):
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
            savecheckpoint(model, os.path.join(savepath, "bestcheckpoint.pt"))
            with open(os.path.join(savepath, "bestmodel.txt"), "w") as f:
                f.write(str(bestaccu))
        savecheckpoint(model, os.path.join(savepath, "lastcheckpoint.pt"))
        with open(os.path.join(savepath, "lastmodel.txt"), "w") as f:
            f.write(str(accu))

all_labels_ = [i.item() for i in all_labels]
all_predictions_ = [i.item() for i in all_predictions]
print(classification_report(all_labels_, all_predictions_))
# %%
