import torch
from torchsummary import summary
from ganslate.configs import base
from ganslate.nn.discriminators.patchgan.patchgan3d import PatchGAN3D
from ganslate.nn.generators import Vnet3D
from torchviz import make_dot


discriminator = PatchGAN3D(1, 64, 2, (4, 4, 4), 'instance')
generator = Vnet3D(1, 1, 'instance')
# summary(discriminator, (1,64,64,64), device='cpu')
# summary(generator, (1,64,64,64), device='cpu')
summary(discriminator, (1, 32, 160, 160))
summary(generator, (1, 32, 160, 160))
# torch.onnx.export(discriminator, torch.randn(1, 32, 160, 160), '/mnt/homeGPU/tenayat/other_data/patchgan3d.onnx')

x = torch.randn(1, 32, 160, 160)
y = discriminator(x)
make_dot(y, params=dict(discriminator.named_parameters())).render("/mnt/homeGPU/tenayat/other_data/discriminator_graph", format="png")
# make_dot(y.mean(), params=dict(model.named_parameters()))