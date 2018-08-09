import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *

class FaF(nn.Module):
    """ FaF is a 3D version of SSD
    By default, 5 Frames are stacked up to create 3D tensors.
    And 2 Conv3D layers without temporal padding are applied for
    temporal dimension fusion. (Later fusion)
    
    Like other trackers we only have 2 classes for each anchor
    here: object and background.

    Args:
        phase: (stirng) can be "test" or "train"
        size: input size [width, height, frame]
    """

    def __init__(self, phase, size, base, extras, head, cfg):
        super(FaF, self).__init__()
        self.phase = phase
        self.size = size

        self.priorbox = PriorBox(self.cfg)
        self.anchors = torch.Tensor(self.priorbox.forward())

        # FaF network
        self.vgg = nn.ModuleList(base)
        self.L2Norm = L2Norm(256, 2)

        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            # self.softmax = nn.Softmax(dim=-1)
            # self.detect = Detect(2, 0, 200, 0.01, 0.45)
            pass

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        # apply vgg - go to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # finish vgg
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(s)

        # apply extras
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to sources
        # permute loc shape to [layer, batch_size, h, w, (class / loc) * num_anchors * num_frames]
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # shape: [batch_size, -1]
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # TODO : output
        if self.phase == "test":
            pass
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, 2),
                self.anchors
            )
        
        return output

def vgg(cfg, i=3, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=1)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)]
        else:
            values = v.split(str="-")
            conv = values[0]
            out_channels = int(values[1])

            if conv == '2':
                conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
            else:
                # Conv3d w/o temporal padding, so temporal channel will reduce by 2
                conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=(0, 1, 1))
                if batch_norm:
                    layers += [conv3d, nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True)]
                else:
                    layers += [conv3d, nn.ReLU(inplace=True)]
            
            in_channels = out_channels
    
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return layers

def add_extras(cfg, i, batch_norm=False):
    # feature scaling extra layers
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])] # this has no padding
            flag = not flag
        in_channels = v
    
    return layers

def multibox(vgg, extra_layers, cfg, num_frames, num_classes):
    # classifiers - output detection for current frame, and predictions for next 4 frames
    loc_layers = []
    conf_layers = []
    vgg_source = [-2] # we are visiting out_channels so should point to CNN layer

    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4 * num_frames, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes * num_frames, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], len(vgg_source)):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4 * num_frames, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes * num_frames, kernel_size=3, padding=1)]
    
    return vgg, extra_layers, (loc_layers, conf_layers)

base = ['2-64', '2-64', 'M', '3-128', '2-128', 'M', '3-256', '2-256', '2-256', 'C', '2-512', '2-512', '2-512', 'M', '2-512', '2-512', '2-512']
extras = [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 256]
mbox = [4, 6, 6, 6, 6, 4]

def build_faf(phase='train', size=[300, 300, 5], num_classes=30, cfg={}):
    base_, extras_, head_ = multibox(
        vgg(base, 3),
        add_extras(extras, 1024),
        mbox,
        size[2],
        num_classes
    )

    return FaF(phase, size, base_, extras_, head_, cfg)
