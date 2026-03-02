import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embedding_size, in_channels=3, model_name='resnet50'):
        super(EncoderCNN, self).__init__()
        # select model
        model_constructor = getattr(models, model_name)
        resnet = model_constructor(pretrained=True)
        
        # Modify the first convolutional layer to handle different input channels
        if in_channels != 3:
            original_first_conv = resnet.conv1
            # Create new first conv layer with desired input channels
            new_first_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=original_first_conv.out_channels,
                kernel_size=original_first_conv.kernel_size,
                stride=original_first_conv.stride,
                padding=original_first_conv.padding,
                bias=original_first_conv.bias
            )
            
            # For >3 channels, you might want to repeat or use custom initialization
            # Here we just randomly initialize (you might want a better approach)
            nn.init.kaiming_normal_(new_first_conv.weight, mode='fan_out', nonlinearity='relu')
            resnet.conv1 = new_first_conv

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embedding_size)

    def forward(self, images):
        if len(images.shape) == 3:
            images = images.unsqueeze(1)
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    
if __name__ == "__main__":
    encoder = EncoderCNN(embedding_size=128, in_channels=32)
    images = torch.randn(1, 32, 128, 32)
    features = encoder(images)
    print(features.shape)