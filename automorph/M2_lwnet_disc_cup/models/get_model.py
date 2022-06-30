import sys
from .res_unet_adrian import UNet as unet

import torch

# from .res_unet_adrian import WNet as wnet
class wnet(torch.nn.Module):
    def __init__(self, n_classes=1, in_c=3, layers=(8,16,32), conv_bridge=True, shortcut=True, mode='train'):
        super(wnet, self).__init__()
        self.unet1 = unet(in_c=in_c, n_classes=n_classes, layers=layers, conv_bridge=conv_bridge, shortcut=shortcut)
        self.unet2 = unet(in_c=in_c+n_classes, n_classes=n_classes, layers=layers, conv_bridge=conv_bridge, shortcut=shortcut)
        self.n_classes = n_classes
        self.mode=mode

    def forward(self, x):
        x1 = self.unet1(x)
        x2 = self.unet2(torch.cat([x, x1], dim=1))
        if self.mode!='train':
            return x2
        return x1,x2


def get_arch(model_name, in_c=3, n_classes=1):

    if model_name == 'unet':
        model = unet(in_c=in_c, n_classes=n_classes, layers=[8,16,32], conv_bridge=True, shortcut=True)
    elif model_name == 'big_unet':
        model = unet(in_c=in_c, n_classes=n_classes, layers=[12,24,48], conv_bridge=True, shortcut=True)
    elif model_name == 'wnet':
        model = wnet(in_c=in_c, n_classes=n_classes, layers=[8,16,32], conv_bridge=True, shortcut=True)
    elif model_name == 'big_wnet':
        model = wnet(in_c=in_c, n_classes=n_classes, layers=[8,16,32,64], conv_bridge=True, shortcut=True)


    else: sys.exit('not a valid model_name, check models.get_model.py')

    return model
if __name__ == '__main__':
    import time
    batch_size = 1
    batch = torch.zeros([batch_size, 1, 80, 80], dtype=torch.float32)
    model = get_arch('unet')
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('Forward pass (bs={:d}) when running in the cpu:'.format(batch_size))
    start_time = time.time()
    logits = model(batch)
    print("--- %s seconds ---" % (time.time() - start_time))

