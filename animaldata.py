#%%
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, Compose
class Animal10data(Dataset):
    def __init__(self, root, train): #root=data/animalsv2
        self.transform = Compose([Resize((224,224)), ToTensor()])
        self.root = root
        self.train = train
        path_train = os.path.join(root,'train')
        path_test = os.path.join(root,'test')
        category_train = os.listdir(path_train)
        category_test = os.listdir(path_test)
        if category_train != category_test:
            raise ValueError("Train and test set have different categories")
        else:
            self.categories = category_train
            self.categories.sort()
        if train:
            self.path = path_train
        else:
            self.path = path_test
        self.class_to_idx = {name: i for i, name in enumerate(self.categories)}

        self.img_paths = []
        for category in self.categories:
            path_to_category = os.path.join(self.path, category) #path = ata/animalsv2/train_or_test/category
            for file in os.listdir(path_to_category):
                path = os.path.join(path_to_category, file)
                self.img_paths.append((path, self.class_to_idx[category]))

    def img(self,index):
        img_path, _ = self.img_paths[index]
        img = Image.open(img_path)
        return img

    def __getitem__(self, index):
        img_path, label = self.img_paths[index]
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = self.transform(img)
        return img , label

    def __len__(self):
        return len(self.img_paths)

    def getitem_path(self, index):
        path, label = self.img_paths[index]
        return path , label

if __name__ == "__main__":
    trainset = Animal10data(root='data/animalsv2', train=True)
    testset = Animal10data(root='data/animalsv2', train=False)
# %%
