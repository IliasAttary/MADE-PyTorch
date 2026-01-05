import torch
import torchvision
import torchvision.transforms as transforms 
import numpy as np
import matplotlib.pyplot as plt

class Binarize():
    def __call__(self, tensor):
        return (tensor > 0.5).float()
    
class Dataloader():
    def __init__(self, dir='./data', dataset='BiMNIST', batch_size=128):
        
        self.dir = dir
        self.dataset = dataset
        self.batch_size = batch_size

        # Binary MNIST
        if self.dataset == 'BiMNIST':
            transform = transforms.Compose(
                            [transforms.ToTensor(), 
                            Binarize(),
                            transforms.Lambda(lambda x: x.view(-1))]) # Flatten 28x28 images to 784 vector
            
            self.trainset = torchvision.datasets.MNIST(root=self.dir, train=True,
                                        download=True, transform=transform)
            self.testset = torchvision.datasets.MNIST(root=self.dir, train=False,
                                       download=True, transform=transform)


        # Data loaders (MNIST: (B, 784) per itration)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size,
                                          shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size,
                                         shuffle=False)
        
    def get_loaders(self, visualize=False):
        if visualize:
            self.visualize(self.trainloader)
        return self.trainloader, self.testloader
    
    def visualize(self, dataloader, num_images=8):
        images, labels = next(iter(dataloader))
        num_images = min(num_images, images.size(0))

        fig, axes = plt.subplots(1, num_images, figsize=(num_images*2, 2))
        if num_images == 1:
            axes = [axes]

        for i in range(num_images):
            img = images[i].detach().cpu().view(28, 28).numpy()
            axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[i].set_title(f"{int(labels[i])}")
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()



            

        
        

