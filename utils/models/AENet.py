import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from collections import OrderedDict


######################################################################
# def weights_init_kaiming(m):
#     classname = m.__class__.__name__
#     # print(classname)
#     if classname.find('Conv2d') != -1:
#         init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
#         init.constant(m.bias.data, 0.0)
#     elif classname.find('Linear') != -1:
#         init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
#         # init.constant(m.bias.data, 0.0)
#     elif classname.find('BatchNorm1d') != -1:
#         init.normal(m.weight.data, 1.0, 0.02)
#         init.constant(m.bias.data, 0.0)
######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        # init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        # init.constant(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        # add_block = []
        add_block1 = []
        add_block2 = []
        add_block1 += [nn.BatchNorm1d(input_dim)]
        if relu:
            add_block1 += [nn.LeakyReLU(0.1)]
        add_block1 += [nn.Linear(input_dim, num_bottleneck, bias=False)]
        add_block2 += [nn.BatchNorm1d(num_bottleneck)]

        # add_block = nn.Sequential(*add_block)
        # add_block.apply(weights_init_kaiming)
        add_block1 = nn.Sequential(*add_block1)
        add_block1.apply(weights_init_kaiming)
        add_block2 = nn.Sequential(*add_block2)
        add_block2.apply(weights_init_kaiming)
        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num, bias=False)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block1 = add_block1
        self.add_block2 = add_block2
        self.classifier = classifier

    def forward(self, x):

        if len(x.size()) == 1:
            x = torch.unsqueeze(x,dim=0)

        x = self.add_block1(x)
        x = self.add_block2(x)
        x = self.classifier(x)
        return x


# ft_net_50_1
class ft_net(nn.Module):
    def __init__(self, num_classes):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1, 1)
        self.model.layer4[0].conv2.stride = (1, 1)
        self.avgpool_1 = nn.AdaptiveAvgPool2d((1, 1))
        # self.avgpool_2 = nn.AdaptiveAvgPool2d((2,2))

        self.avgpool_2 = nn.AdaptiveAvgPool2d((2, 2))
        self.avgpool_3 = nn.AdaptiveMaxPool2d((2, 2))
        self.avgpool_4 = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool_5 = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier_1 = ClassBlock(1024, num_classes, num_bottleneck=512)
        self.classifier_2 = ClassBlock(2048, num_classes, num_bottleneck=512)
        self.classifier_3 = ClassBlock(8192, num_classes, num_bottleneck=512)

        self.local_fc = nn.ModuleList()


    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x0 = self.model.layer3(x)
        x = self.model.layer4(x0)
        x3 = self.model.avgpool(x)
        x_3 = self.avgpool_5(x)
        x_41 = self.avgpool_2(x)
        x_4 = self.avgpool_3(x)
        x_0 = self.avgpool_1(x0)
        x_1 = self.avgpool_4(x0)
        x0 = x_0 + x_1
        x_31 = x3 + x_3
        x4 = x_41 + x_4
        #
        x6 = torch.squeeze(x0)
        x_0 = torch.squeeze(x_0)
        x_1 = torch.squeeze(x_1)
        x3 = torch.squeeze(x3)
        x_3 = torch.squeeze(x_3)

        # print(x0.shape,x_31.shape,x4.shape)
        # ((32, 1024, 1, 1), (32, 2048, 1, 1), (32, 2048, 2, 2))
        # x7 = x1.view(x1.size(0),-1)

        #
        x9 = torch.squeeze(x_31)
        x_10 = x_4.view(x_4.size(0), -1)
        x_11 = x_41.view(x_41.size(0), -1)
        x10 = x4.view(x4.size(0), -1)

        _x10 = x10.unsqueeze(dim=2).unsqueeze(dim=2)

        sorted_x0 = x0.sort(dim=1,descending=True)[0]
        sorted_x_31 = x_31.sort(dim=1,descending=True)[0]
        sorted_x10 = _x10.sort(dim=1,descending=True)[0]

        #
        x16 = self.classifier_1(x6)
        x18 = self.classifier_2(x9)
        x22 = self.classifier_3(x10)
        #
        return x16, x18, x22, x_0, x_1, x3, x_3, x_10, x_11, torch.cat([sorted_x0[:,:256,:,:],sorted_x_31[:,:768,:,:],sorted_x10[:,:2048,:,:]],dim=1)


class Decoder(nn.Module):

    def __init__(self,inplanes):

        super(Decoder,self).__init__()

        self.deconv5 = nn.Sequential(OrderedDict([
            ('deconv',nn.ConvTranspose2d(inplanes, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(64)),
            ('relu', nn.LeakyReLU(True))
        ]))

        self.deconv4 = nn.Sequential(OrderedDict([
            ('deconv', nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(32)),
            ('relu', nn.LeakyReLU(True))
        ]))

        self.deconv3 = nn.Sequential(OrderedDict([
            ('deconv', nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(32)),
            ('relu', nn.LeakyReLU(True))
        ]))

        self.deconv2 = nn.Sequential(OrderedDict([
            ('deconv', nn.ConvTranspose2d(16, 8, kernel_size=3, stride=4, padding=0, output_padding=1, bias=True)),
            ('bn', nn.InstanceNorm2d(32)),
            ('relu', nn.LeakyReLU(True))
        ]))

        self.deconv1 = nn.Sequential(OrderedDict([
            ('pad', nn.ReflectionPad2d(3)),
            ('conv', nn.Conv2d(8, 3, kernel_size=7, stride=1, padding=0, bias=False)),
            ('tanh', nn.Tanh())
        ]))

        self.apply(weights_init_kaiming)


    def forward(self,x):

        x0 = self.deconv5(x)

        x1 = self.deconv4(x0)

        x2 = self.deconv3(x1)

        x3 = self.deconv2(x2)

        x4 = self.deconv1(x3)

        return x4

class MB_Decoder(nn.Module):

    def __init__(self,inplanes=128):
        super(MB_Decoder,self).__init__()

        for i in range(24):
            setattr(self,'Decoder{}'.format(i),Decoder(inplanes=inplanes))


    def forward(self,x):

        outputs = []

        for i in range(24):

            outputs.append(getattr(self,'Decoder{}'.format(i))(x[:,i * 128:(i+1) * 128,:,:]))

        return outputs

class AutoEncoder(nn.Module):
    def __init__(self,num_classes):
        super(AutoEncoder,self).__init__()

        self.encoder = ft_net(num_classes=num_classes)
        self.local_decoder = MB_Decoder(inplanes=128)
        # self.local_decoder = nn.Sequential()

    def forward(self,x):

        x16, x18, x22, x_0, x_1, x3, x_3, x_10, x_11,codes = self.encoder(x)
        # print(codes.shape)
        dsaps = self.local_decoder(codes)
        outputs = [x16, x18, x22, x_0, x_1, x3, x_3, x_10, x_11, dsaps]

        # print([x.shape for x in dsaps])
        return outputs



def AENet(num_classes, loss, pretrained = True, ** kwargs):
    model = AutoEncoder(num_classes=num_classes)
    return model