from torchvision import transforms, datasets


def load_dataset():

    transform = transforms.Compose([transforms.Resize([105, 78]),
                                    transforms.CenterCrop(size=[60, 30]),
                                    transforms.ToTensor()])

    train_path = r'WF-data/train'
    train_dataset = datasets.ImageFolder(train_path, transform=transform, target_transform=None)

    test_path = r'WF-data/test'
    test_dataset = datasets.ImageFolder(test_path, transform=transform, target_transform=None)

    return train_dataset, test_dataset


# train_dataset, test_dataset = load_dataset()
# print(len(train_dataset), len(test_dataset))
