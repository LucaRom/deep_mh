import torch
from torch import nn
from torch.nn import functional as F


class UNet_multi(nn.Module):
    """
    Paper: `U-Net: Convolutional Networks for Biomedical Image Segmentation
    <https://arxiv.org/abs/1505.04597>`_
    Paper authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox
    Implemented by:
        - `Annika Brundyn <https://github.com/annikabrundyn>`_
        - `Akshay Kulkarni <https://github.com/akshaykvnit>`_
    Args:
        num_classes: Number of output classes required
        input_channels: Number of channels in input images (default 3)
        num_layers: Number of layers in each side of U-net (default 5)
        features_start: Number of features in first layer (default 64)
        bilinear: Whether to use bilinear interpolation or transposed convolutions (default) for upsampling.
    """

    def __init__(
        self,
        num_classes: int,
        input_channels: int = 3,
        input_channels_lidar: int = 3,
        num_layers: int = 5,
        features_start: int = 64,
        bilinear: bool = False,
    ):

        if num_layers < 1:
            raise ValueError(f"num_layers = {num_layers}, expected: num_layers > 0")

        super().__init__()
        self.num_layers = num_layers

        # layers = [DoubleConv(input_channels, features_start)]
        # #layers_lidar = [DoubleConv(input_channels_lidar, features_start)]

        # feats = features_start
        # for _ in range(num_layers - 1):
        #     layers.append(Down(feats, feats * 2))
        #     feats *= 2

        # # One input encoder
        # # for _ in range(num_layers - 1):
        # #     layers.append(Up(feats, feats // 2, bilinear))
        # #     feats //= 2

        # # Two input encoders
        # feats = feats * 2
        # for _ in range(num_layers - 1):
        #     layers.append(Up(feats, feats // 2, bilinear))
        #     feats //= 2

        # layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))
        # #layers_lidar.append(nn.Conv2d(feats, num_classes, kernel_size=1))

        # self.layers = nn.ModuleList(layers)

        def build_down_layers(input_channels, features_start, num_layers):
            layers = [DoubleConv(input_channels, features_start)]

            feats = features_start
            for _ in range(num_layers - 1):
                layers.append(Down(feats, feats * 2))
                feats *= 2

            #self.layers = nn.ModuleList(layers)
            layers = nn.ModuleList(layers)

            return layers

        def build_up_layers(num_layers, bilinear, num_classes):
            # Two input encoders
            layers = []

            feats = 1024
            feats = feats * 2
            for _ in range(num_layers - 1):
                layers.append(Up(feats, feats // 2, bilinear))
                feats //= 2

            layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))
            #layers_lidar.append(nn.Conv2d(feats, num_classes, kernel_size=1))

            layers = nn.ModuleList(layers)
            
            return layers

        self.conv_lidar = build_down_layers(input_channels_lidar, features_start, num_layers)
        self.conv_opt = build_down_layers(input_channels, features_start, num_layers)
        self.up = build_up_layers(num_layers, bilinear, num_classes)

    # # One input encoder
    # def forward(self, x):
    #     xi = [self.layers[0](x)]
    #     # Down path
    #     for layer in self.layers[1 : self.num_layers]:
    #         xi.append(layer(xi[-1])) # Sen2

    #     # Up path
    #     for i, layer in enumerate(self.layers[self.num_layers : -1]):
    #         xi[-1] = layer(xi[-1], xi[-2 - i])
    #     return self.layers[-1](xi[-1])

    # Two inputs encoders
    def forward(self, x, y):
        xi = [self.conv_opt[0](x)]
        yi = [self.conv_lidar[0](y)]

        for layer in self.conv_opt[1 : self.num_layers]:
            xi.append(layer(xi[-1])) # Sen2

        for layer in self.conv_lidar[1 : self.num_layers]:
            yi.append(layer(yi[-1])) # LiDAR

        # Up path
        xi[-1] = torch.cat([xi[-1], yi[-1]], dim=1)
        #for i, layer in enumerate(self.up[self.num_layers : -1]):
        for i, layer in enumerate(self.up[ : -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i], yi[-2 - i])

        return self.up[-1](xi[-1])

    # def forward(self, x, y):
    #     xi = [self.layers[0](x)]
    #     yi = [self.layers_lidar[0](y)]
    #     # Down path
    #     for layer in self.layers[1 : self.num_layers]:
    #         xi.append(layer(xi[-1])) # Sen2

    #     for layer in self.layers_lidar[1 : self.num_layers]:
    #         yi.append(layer(yi[-1])) # Sen2
            
    #     # Up path
    #     xi[-1] = torch.cat([xi[-1], yi[-1]], dim=1)
    #     for i, layer in enumerate(self.layers[self.num_layers : -1]):
    #         xi[-1] = layer(xi[-1], xi[-2 - i], yi[-2 - i])
    #     return self.layers[-1](xi[-1])


class DoubleConv(nn.Module):
    """[ Conv2d => BatchNorm (optional) => ReLU ] x 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """Downscale with MaxPool => DoubleConvolution block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    """Upsampling (by either bilinear interpolation or transpose convolutions) followed by concatenation of feature
    map from contracting path, followed by DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    # One input encoder
    # def forward(self, x1, x2):
    #     x1 = self.upsample(x1)

    #     # Pad x1 to the size of x2
    #     diff_h = x2.shape[2] - x1.shape[2]
    #     diff_w = x2.shape[3] - x1.shape[3]

    #     x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

    #     # Concatenate along the channels axis
    #     x = torch.cat([x2, x1], dim=1)
    #     return self.conv(x)

    # Two input encoders
    def forward(self, x1, x2, y2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([y2, x2, x1], dim=1)
        return self.conv(x)

def test():
    x = torch.randn((1, 3, 161, 161))
    y = torch.randn((1, 5, 161, 161))
    model = UNet_multi(
            num_classes=3,
            input_channels=3,
            input_channels_lidar=5,
            num_layers=5,
            features_start=64,
            bilinear=True
            )

    dot = make_dot(model(x,y), params=dict(model.named_parameters()))
    dot.format = 'png'
    dot.render('unet_multi')


    # One input
    #preds = model(x)

    # Two inputs
    # preds = model(x, y)

    # print(preds.shape, x.shape)
    # assert preds.shape == x.shape
    # print(preds)
    
    #print(model)

if __name__ == "__main__":

    from torchsummary import summary
    from torchviz import make_dot, make_dot_from_trace

    import os
    os.environ["PATH"] += os.pathsep + 'E:/Program Files/Graphviz/bin/'
 
    test()

    # batch_size = 2
    # num_classes = 5  # one hot
    # initial_kernels = 32

    # net = Multi_Unet(1, num_classes, initial_kernels)
    # print("total parameter:" + str(netSize(net)))  # 2860,0325
    # # torch.save(net.state_dict(), 'model.pth')
    # MRI = torch.randn(batch_size, 4, 64, 64)    # Batchsize, modal, hight,

    # if torch.cuda.is_available():
    #     net = net.cuda()
    #     MRI = MRI.cuda()

    # segmentation_prediction = net(MRI)
    # print(segmentation_prediction.shape)

    #summary(model.cuda(), [(3, 161, 161)])

 
