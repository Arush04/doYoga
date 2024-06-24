import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

N_images = 10

# Defining the transformation for the dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    ])

# Loading the dataset
train_dataset = datasets.ImageFolder(root='./DATASET/TRAIN', transform=transform)

# Creating a Dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=N_images,shuffle=True)

dataiter = iter(train_loader)
images, labels = next(dataiter)

# Class names
classes = train_dataset.classes

# Plot Images
fig, ax = plt.subplots(1, N_images, figsize=(17, 7))

for i in range(N_images):
    # Convert the tensor to a numpy array and transpose to (H, W, C)
    im = images[i].numpy().transpose((1, 2, 0))
    lbl = labels[i].item()

    # Display the image
    ax[i].imshow(im, interpolation='bilinear')
    ax[i].set_title(f'{classes[lbl]}')
    ax[i].axis('off')

plt.show()
