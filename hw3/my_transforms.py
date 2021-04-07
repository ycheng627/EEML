import torchvision.transforms as transforms
from PIL import Image


def get_train_tfm():
    return transforms.Compose([
        # Resize the image into a fixed shape (height = width = 224)
        # transforms.Resize((224, 224)),
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=30, translate=(0.2, 0.2)),
        # You may add some transforms here.
        # ToTensor() should be the last one of the transforms.
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # imagenet policy

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
def get_test_tfm():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])