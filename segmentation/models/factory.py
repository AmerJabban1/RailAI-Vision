# segmentation/models/factory.py

import segmentation_models_pytorch as smp
from segmentation.models.unet_vanilla import UNet

def get_model(name, in_channels=3, out_channels=19, pretrained=True):
    if name == "UNet-Vanilla":
        return UNet(in_channels, out_channels)
    elif name == "UNet-Pretrained":
        return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet" if pretrained else None, in_channels=in_channels, classes=out_channels)
    elif name == "DeepLabV3+":
        return smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet" if pretrained else None, in_channels=in_channels, classes=out_channels)
    elif name == "FPN":
        return smp.FPN(encoder_name="resnet34", encoder_weights="imagenet" if pretrained else None, in_channels=in_channels, classes=out_channels)
    elif name == "PSPNet":
        return smp.PSPNet(encoder_name="resnet50", encoder_weights="imagenet" if pretrained else None, in_channels=in_channels, classes=out_channels)
    else:
        raise ValueError(f"Unknown model name: {name}")