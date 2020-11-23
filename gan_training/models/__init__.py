from gan_training.models import (
    resnet4,
    resnet4_AdaFM_bias,
    resnet4_AdaFM_bias_classCondition,
    resnet4_AdaFM_accumulate_multitasks,
)

generator_dict = {
    'resnet4': resnet4.Generator,
    'resnet4_AdaFM_bias': resnet4_AdaFM_bias.Generator,
    'resnet4_AdaFM_bias_classCondition': resnet4_AdaFM_bias_classCondition.Generator,
    'resnet4_AdaFM_accumulate_multitasks': resnet4_AdaFM_accumulate_multitasks.Generator,
}

discriminator_dict = {
    'resnet4': resnet4.Discriminator,
    'resnet4_AdaFM_bias': resnet4_AdaFM_bias.Discriminator,
    'resnet4_AdaFM_bias_classCondition': resnet4_AdaFM_bias_classCondition.Discriminator,
    'resnet4_AdaFM_accumulate_multitasks': resnet4_AdaFM_accumulate_multitasks.Discriminator,
}

encoder_dict = {
}
