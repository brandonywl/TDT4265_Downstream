import torch
import torch.nn as nn
from typing import Tuple, List


class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), padding=1, stride=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels*2, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels*4, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.LeakyReLU(),
            nn.BatchNorm2d(in_channels*4),
            nn.Conv2d(in_channels=in_channels*4, out_channels=in_channels, kernel_size=(1,1), padding=0),
            nn.LeakyReLU()
        )

        self.skip_resizer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), padding=1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        output = self.model(x)
        output = output + x
        output = self.skip_resizer(output)
        return output


class BasicImprovedModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        kernel_size = (3, 3)


        feat_map1 = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=kernel_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            ResBlock(32, 64),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(in_channels=64, out_channels=output_channels[0], kernel_size=kernel_size, stride=2, padding=1),
            nn.LeakyReLU()
        )

        feat_map2 = nn.Sequential(
            nn.LeakyReLU(),
            ResBlock(output_channels[0], 128),
            nn.Conv2d(in_channels=128, out_channels=output_channels[1], kernel_size=kernel_size, stride=2, padding=1),
            nn.LeakyReLU()
        )

        feat_map3 = nn.Sequential(
            nn.LeakyReLU(),
            ResBlock(in_channels=output_channels[1], out_channels=256, kernel_size=(5,5), stride=1, padding=2),
            nn.Conv2d(in_channels=256, out_channels=output_channels[2], kernel_size=kernel_size, stride=2, padding=1),
            nn.LeakyReLU()
        )

        feat_map4 = nn.Sequential(
            nn.LeakyReLU(),
            ResBlock(in_channels=output_channels[2], out_channels=128, kernel_size=(5,5), stride=1, padding=2),
            nn.Conv2d(in_channels=128, out_channels=output_channels[3], kernel_size=kernel_size, stride=2, padding=1),
            nn.LeakyReLU()
        )
        
        feat_map5 = nn.Sequential(
            nn.LeakyReLU(),
            ResBlock(in_channels=output_channels[3], out_channels=128, kernel_size=kernel_size, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=output_channels[4], kernel_size=kernel_size, stride=2, padding=1),
            nn.LeakyReLU()
        )
        
        feat_map6 = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=output_channels[4], out_channels=128, kernel_size=kernel_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=output_channels[5], kernel_size=kernel_size, stride=1, padding=0),
            nn.LeakyReLU()
        )

        feature_extractors = [
            feat_map1, feat_map2, feat_map3, feat_map4, feat_map5, feat_map6
        ]

        self.feature_extractors = nn.ModuleList(feature_extractors)
        
        

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []

        for feature_extractor in self.feature_extractors:
            x = feature_extractor(x)
            out_features.append(x)


        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

