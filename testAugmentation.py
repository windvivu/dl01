from PIL import Image
from torchvision.transforms import RandomAffine, ColorJitter, RandomRotation, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomResizedCrop, RandomGrayscale, RandomPerspective, RandomErasing

img = Image.open('img/cat.jpeg')

aumentation = RandomAffine(
    degrees=(-5,5), # xoay ảnh
    translate=(0.15,0.15), # dịch ảnh
    scale=(0.85,1.5), # zoom tỉ lệ ảnh
    shear=10, # cắt ảnh
    )

for i in range(10):
    img2 = aumentation(img)
    img2.save(f'img/cat({i}).jpeg')