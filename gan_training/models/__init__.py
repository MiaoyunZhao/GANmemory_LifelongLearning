from gan_training.models import (
    resnet4,
    resnet4_AdaFM_bias,
)

generator_dict = {
    'resnet4': resnet4.Generator,
    'resnet4_AdaFM_bias': resnet4_AdaFM_bias.Generator,
}

discriminator_dict = {
    'resnet4': resnet4.Discriminator,
    'resnet4_AdaFM_bias': resnet4_AdaFM_bias.Discriminator,
}

encoder_dict = {
}