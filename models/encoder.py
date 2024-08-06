import torch
import torch.nn as nn

class SimpleEncoder(nn.Module):
    def __init__(self):
        super(SimpleEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Output size: (64, 64, 64)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output size: (128, 32, 32)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output size: (256, 16, 16)
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # Output size: (512, 8, 8)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 512),  # Adjust this based on final feature map size
            nn.ReLU(),
            nn.Linear(512, 256)  # Adjust the output size based on your GAN's latent space
        )

    def forward(self, x):
        print(f'Input shape: {x.shape}')
        x = self.encoder[0](x)
        print(f'After conv1: {x.shape}')
        x = self.encoder[1](x)
        x = self.encoder[2](x)
        print(f'After conv2: {x.shape}')
        x = self.encoder[3](x)
        x = self.encoder[4](x)
        print(f'After conv3: {x.shape}')
        x = self.encoder[5](x)
        x = self.encoder[6](x)
        print(f'After conv4: {x.shape}')
        x = self.encoder[7](x)
        print(f'After flatten: {x.shape}')
        x = self.encoder[8](x)
        print(f'After fc1: {x.shape}')
        x = self.encoder[9](x)
        print(f'Output shape: {x.shape}')
        return x
