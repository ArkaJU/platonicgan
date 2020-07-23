from scripts.models import resnet, default, vnet

# add different models here

generator_dict = {
    'default': default.Generator,
    'vnet': vnet.VNet_Generator
}

discriminator_dict = {
    'default': default.Discriminator,
    'default_3d': default.Discriminator3d
}

encoder_dict = {
    'default': default.Encoder,
    'vnet': vnet.VNet_Encoder
}
