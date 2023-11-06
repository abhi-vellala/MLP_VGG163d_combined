import torch
import torch.nn as nn
import torch.nn.functional as F
import json


class MLPVGG163DAdjusted(nn.Module):
    def __init__(self, model_configs, num_classes, device):
        super(MLPVGG163DAdjusted, self).__init__()
        self.device = device
        self.model_configs = model_configs
        self.num_classes = num_classes
        self.convnet = self.ConvBlock().to(device=self.device)
        self.densenet = self.DenseBlock().to(device=device)
        self.mlpfeatures = self.MLPFeatures().to(device=device)
        self.mlpcombined = self.MLPCombined().to(device=device)


    def ConvBlock(self):

        convblock = self.model_configs['convblock']
        conv_layers = []

        for num in range(len(convblock)):
            layer_configs = convblock[str(num)]
            # print(num)
            # print(layer_configs)
            layer_name = layer_configs['name']
            try:
                in_channels = layer_configs['in_channels']
                out_channels = layer_configs['out_channels']
                kernel_size = layer_configs['kernel_size']
                stride = layer_configs['stride']
                padding = layer_configs['padding']
                batch_norm = layer_configs['batch_norm']
                relu = layer_configs['relu']
            except:
                kernel_size = layer_configs['kernel_size']
                stride = layer_configs['stride']

            if layer_name == 'cnn':
                layer = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)]
                if batch_norm:
                    layer.append(nn.BatchNorm3d(out_channels))
                if relu:
                    layer.append(nn.ReLU())

            if layer_name == 'maxpool':
                layer = [nn.MaxPool3d(kernel_size=kernel_size, stride=stride)]

            conv_layers += layer
        
        return nn.Sequential(*conv_layers)
    
    def DenseBlock(self):

        denseblock = self.model_configs['denseblock']
        dense_layers = []

        for num in range(len(denseblock)):
            layer_configs = denseblock[str(num)]
            out_features = layer_configs['out_features']
            layer = []
            if num == 0:
                if 'drop_out' in layer_configs.keys():
                    layer.append(nn.Dropout(layer_configs['drop_out']))

                layer.append(nn.LazyLinear(out_features=out_features))

            if num > 0:
                in_features = layer_configs['in_features']
                if 'drop_out' in layer_configs.keys():
                    layer.append(nn.Dropout(layer_configs['drop_out']))
    
                layer.append(nn.Linear(in_features,out_features))

            if 'relu' in layer_configs.keys():
                if layer_configs['relu']:
                    layer.append(nn.ReLU())

            dense_layers += layer

        return nn.Sequential(*dense_layers)
    
    def MLPFeatures(self):
        return nn.Sequential(
            nn.LazyLinear(out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=20)
        )
    
    def MLPCombined(self):
        return nn.Sequential(
            nn.LazyLinear(out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=self.num_classes),
        )


    def forward(self, x, y):

        for i, layer in enumerate(self.convnet):
            print(i)
            x = layer(x)
            # print(f"{layer}: {x.shape}")

        x = torch.flatten(x,1)
        x = x.reshape(x.size(0), -1)

        for i, layer in enumerate(self.densenet):
            print(i)
            x = layer(x)
            # print(f"{layer}: {x.shape}")


        for i, layer in enumerate(self.mlpfeatures):
            print(i)
            y = layer(y)
            # print(f'{layer}: {y.shape}')

        xcombined = torch.cat((x,y),1)

        # print(f'Xcombined shape: {xcombined.shape}')

        for i,layer in enumerate(self.mlpcombined):
            print(i)
            xcombined = layer(xcombined)
            print(f'{layer}: {xcombined.shape}')

        return xcombined
    
    

if __name__ == "__main__":

    with open('vgg16_adj_configs.json') as config_file:
        model_configs = json.load(config_file)

    # print(model_configs['convblock']['0'])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = MLPVGG163DAdjusted(model_configs,num_classes=2, device=device)

    x = torch.randn(1,1,24,224,224).to(device=device)
    y = torch.randn(1,2).to(device=device)

    print(f"input Image: {x.shape}")
    print(f'Input Features: {y.shape}')

    x = model(x, y)


