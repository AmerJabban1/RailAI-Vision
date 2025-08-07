import segmentation_models_pytorch as smp

def get_deeplab_model(num_classes, encoder='resnet50', encoder_weights='imagenet'):
    """
    Initializes a DeepLabV3+ model with the specified number of output classes.

    Args:
        num_classes (int): Number of output classes.
        encoder (str): Encoder backbone for DeepLabV3+. Default is 'resnet50'.
        encoder_weights (str or None): Pretrained weights for encoder. Default is 'imagenet'.

    Returns:
        smp.DeepLabV3Plus: Initialized model.
    """
    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )
    return model