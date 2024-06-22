import torch
from torch import nn
import torch.nn.functional as F

class ChannelSpatialAttentionModule(nn.Module):
    def __init__(self, num_features, reduction_ratio=16):
        super(ChannelSpatialAttentionModule, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(num_features // reduction_ratio, num_features, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x) * x

        # Spatial attention
        max_pool = torch.max(ca, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(ca, dim=1, keepdim=True)
        pool = torch.cat([max_pool, avg_pool], dim=1)
        sa = self.spatial_attention(pool) * ca

        return sa

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.attention = ChannelSpatialAttentionModule(out_ch)

    def forward(self, x):
        x = self.double_conv(x)
        x = self.attention(x)
        return x

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, skip_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch + skip_channels, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch + skip_channels, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)  # Concatenation of skip connection and upsampled feature map
        return self.conv(x)

class unet_3enco_concat_attention(nn.Module):
    def __init__(self, num_classes, input_channels, input_channels_lidar, input_channels_radar, features_start=64, bilinear=True):
        super(unet_3enco_concat_attention, self).__init__()
        self.num_classes = num_classes
        self.bilinear = bilinear
        
        # Initialize down/encoder paths for each input type
        self.encoder_opt = [DoubleConv(input_channels, features_start)]
        self.encoder_lidar = [DoubleConv(input_channels_lidar, features_start)]
        self.encoder_radar = [DoubleConv(input_channels_radar, features_start)]

        # Populate encoder layers
        feats = features_start
        for _ in range(4):  # Assuming 4 layers as an example
            self.encoder_opt.append(Down(feats, feats * 2))
            self.encoder_lidar.append(Down(feats, feats * 2))
            self.encoder_radar.append(Down(feats, feats * 2))
            feats *= 2

        # Convert lists to ModuleList
        self.encoder_opt = nn.ModuleList(self.encoder_opt)
        self.encoder_lidar = nn.ModuleList(self.encoder_lidar)
        self.encoder_radar = nn.ModuleList(self.encoder_radar)

        # Define a bottleneck
        self.bottleneck = DoubleConv(feats * 3, feats)

        # Initialize up/decoder path
        self.ups = nn.ModuleList([
            Up(feats, feats // 2, (feats // 2)*3, bilinear),
            Up(feats // 2, feats // 4, (feats // 4)*3, bilinear),
            Up(feats // 4, feats // 8, (feats // 8)*3, bilinear),
            Up(feats // 8, feats // 16, (feats // 16)*3, bilinear)
        ])

        self.final_conv = nn.Conv2d(feats // 16, num_classes, kernel_size=1)

    def forward(self, x_opt, x_lidar, x_radar):
        skips_opt, skips_lidar, skips_radar = [], [], []
        
        for i, layer in enumerate(self.encoder_opt):
            x_opt = layer(x_opt)
            if i < len(self.encoder_opt) - 1:  # Exclude the last layer's output
                skips_opt.append(x_opt)
                
        for i, layer in enumerate(self.encoder_lidar):
            x_lidar = layer(x_lidar)
            if i < len(self.encoder_lidar) - 1:
                skips_lidar.append(x_lidar)
                
        for i, layer in enumerate(self.encoder_radar):
            x_radar = layer(x_radar)
            if i < len(self.encoder_radar) - 1:
                skips_radar.append(x_radar)

        # Merge encoder outputs for upsampling path
        x_concat = torch.cat([x_opt, x_lidar, x_radar], dim=1)  # Concatenate final encoder outputs
        x = self.bottleneck(x_concat)
        
        # Reverse skip connections for proper ordering in upsampling
        skips_opt.reverse()
        skips_lidar.reverse()
        skips_radar.reverse()

        # Up/decoder path
        for i, up in enumerate(self.ups):
            skip = torch.cat([skips_opt[i], skips_lidar[i], skips_radar[i]], dim=1)
            x = up(x, skip)

        return self.final_conv(x)


if __name__ == "__main__" :
    num_classes = 2
    input_channels = 22  # For optical
    input_channels_lidar = 5  # For lidar
    input_channels_radar = 6  # For radar
    model = unet_3enco_concat(num_classes, input_channels, input_channels_lidar, input_channels_radar)

    print(model)