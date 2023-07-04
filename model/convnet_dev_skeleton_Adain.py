import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from model.se_resnet import BottleneckX, SEResNeXt
from model.options import DEFAULT_NET_OPT
from math import sqrt

class MultiPrmSequential(nn.Sequential):
    def __init__(self, *args):
        super(MultiPrmSequential, self).__init__(*args)

    def forward(self, input, cat_feature):
        for module in self._modules.values():
            input = module(input, cat_feature)
        return input
    
def make_secat_layer(block, inplanes, planes, cat_planes, block_count, cur ,depths,stride=1 ):
#     outplanes = planes * block.expansion
    drop_path_rate=0.
    layer_scale_init_value=1e-6
    head_init_scale=1.
    layers = []
    dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, depths)] 
    #     def __init__(self, inplanes, planes, cat_channel, stride=1, downsample=None, drop_rate=0., layer_scale_init_value=1e-6):
        
    for j in range(block_count):
        layers.append(block(inplanes, planes, cat_planes, stride=1, drop_path=dp_rates[cur + j], 
                    layer_scale_init_value=layer_scale_init_value))
        
    return MultiPrmSequential(*layers)


class SeCatLayer(nn.Module):
    def __init__(self, channel, cat_channel, reduction=16):
        super(SeCatLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(cat_channel + channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, cat_feature):
#         print("x.size()")
#         print(x.size())
#         print("cat_feature.size()")
#         print(cat_feature.size())
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = torch.cat([y, cat_feature], 1)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
    
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize   
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
 
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    官方实现的LN是默认对最后一个维度进行的，这里是对channel维度进行的，所以单另设一个类。
    """
 
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
 
 
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
#     *[block(inplanes, planes, cat_planes, stride=1, drop_path=dp_rates[cur + j], 
#                 layer_scale_init_value=layer_scale_init_value) 
    def __init__(self, inplanes, dim, cat_channel, stride=1, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # layer scale
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.selayer = SeCatLayer(dim , cat_channel)
        
    def forward(self, x , cat_feature ):
        shortcut = x
#         print(shortcut.size())
#         b, c = cat_feature.size()
#         cat_feature = cat_feature.view(b, c, 1, 1)

#         _, _, w, h = x.size()
#         cats = cat_feature.expand(b, c, w, h)
#         in_x = torch.cat([x, cats], 1)
        
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        
#         加了一个SE注意力
        b, c, _, _ = x.size()
#         print(x.size())torch.Size([4, 256, 32, 32])
#         print(cat_feature.size()) torch.Size([4, 64])
        x = self.selayer(x , cat_feature)
        x = shortcut + self.drop_path(x)
#         print(x.size())
        return x


class FeatureConv(nn.Module):
    def __init__(self, input_dim=512, output_dim=256, input_size=32, output_size=16, net_opt=DEFAULT_NET_OPT):
        super(FeatureConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.output_size = output_size

        no_bn = not net_opt['bn']
        
        if input_size == output_size * 4:
            stride1, stride2 = 2, 2
        elif input_size == output_size * 2:
            stride1, stride2 = 2, 1
        else:
            stride1, stride2 = 1, 1
        
        seq = []
        seq.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride1, padding=1, bias=False))
        if not no_bn: seq.append(nn.BatchNorm2d(output_dim))
        seq.append(nn.ReLU(inplace=True))
        seq.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=stride2, padding=1, bias=False))
        if not no_bn: seq.append(nn.BatchNorm2d(output_dim))
        seq.append(nn.ReLU(inplace=True))
        seq.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False))
        seq.append(nn.ReLU(inplace=True))

        self.network = nn.Sequential(*seq)

    def forward(self, x):
        return self.network(x)
#   self.deconv1 = DecoderBlock(bottom_layer_len, 4*256, self.color_fc_out, self.layers[0], no_bn=no_bn)
#   一个secat layer + ps

class DecoderBlock(nn.Module):
#              输入通道数、输出通道数
    def __init__(self, inplanes, planes, color_fc_out, block_num, cur,depths):
        super(DecoderBlock, self).__init__()
#         self.secat_layer = make_secat_layer(Block, inplanes, planes, color_fc_out, block_num, cur,depths)
        
#     outplanes = planes * block.expansion
        drop_path_rate=0.
        layer_scale_init_value=1e-6
        head_init_scale=1.
        
        self.down = []
    #     layers.append(block(inplanes, planes, cat_planes, 16, stride, downsample, no_bn=no_bn))
        if inplanes!=planes:
            self.down=nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))
        self.secat_layer = make_secat_layer(Block, planes, planes, color_fc_out, block_num, cur ,depths)
        
        self.ps = nn.PixelShuffle(2)

    def forward(self, x, cat_feature):
        out = self.down(x)
        out = self.secat_layer(out ,cat_feature)
        return self.ps(out)


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out
class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)

def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)
    
# 中间mix的block
class MixBlock(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, style_dim, cardinality=16, stride=1, downsample=None):
        super(MixBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.adain1 = AdaptiveInstanceNorm(planes, style_dim)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, groups=cardinality, bias=False)
        self.adain2 = AdaptiveInstanceNorm(planes, style_dim)

        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.adain3 = AdaptiveInstanceNorm(planes, style_dim)

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, style):
        out = self.conv1(x)
        out = self.adain1(out, style)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.adain2(out, style)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.adain3(out, style)
        out = self.relu(out)

        return out

class Generator(nn.Module):
    def __init__(self, input_size, cv_class_num, iv_class_num, input_dim=1, output_dim=3,
                 layers=[12, 8, 5, 5], net_opt=DEFAULT_NET_OPT):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cv_class_num = cv_class_num
        self.iv_class_num = iv_class_num

        self.input_size = input_size
        self.layers = layers

        self.cardinality = 16

        self.bottom_h = self.input_size // 16
        self.Linear = nn.Linear(cv_class_num, self.bottom_h*self.bottom_h*32)

        self.color_fc_out = 64
        self.net_opt = net_opt

        no_bn = not net_opt['bn']

        if net_opt['relu']:
            self.colorFC = nn.Sequential(
                nn.Linear(cv_class_num, self.color_fc_out), nn.ReLU(inplace=True),
                nn.Linear(self.color_fc_out, self.color_fc_out), nn.ReLU(inplace=True),
                nn.Linear(self.color_fc_out, self.color_fc_out), nn.ReLU(inplace=True),
                nn.Linear(self.color_fc_out, self.color_fc_out)
            )
        else:
            self.colorFC = nn.Sequential(
                nn.Linear(cv_class_num, self.color_fc_out),
                nn.Linear(self.color_fc_out, self.color_fc_out),
                nn.Linear(self.color_fc_out, self.color_fc_out),
                nn.Linear(self.color_fc_out, self.color_fc_out)
            )

        self.conv1 = self._make_encoder_block_first(self.input_dim, 16)
        self.conv2 = self._make_encoder_block(16, 32)
        self.conv3 = self._make_encoder_block(32, 64)
        self.conv4 = self._make_encoder_block(64, 128)
        self.conv5 = self._make_encoder_block(128, 256)
        
        
        #  风格确定 inplanes, planes, style_dim
        self.mix = MixBlock(256 + 256, 256 , 64)
        
        bottom_layer_len = 256 + 64

        self.deconv1 = DecoderBlock(bottom_layer_len, 256*4, self.color_fc_out, self.layers[0], 0 ,sum(layers))
        self.deconv2 = DecoderBlock(256 + 128, 128*4, self.color_fc_out, self.layers[1],self.layers[0],sum(layers))
        self.deconv3 = DecoderBlock(128 + 64, 64*4, self.color_fc_out, self.layers[2], self.layers[1],sum(layers))
        self.deconv4 = DecoderBlock(64 +32, 32*4, self.color_fc_out, self.layers[3], self.layers[2],sum(layers))
        self.deconv5 = nn.Sequential(
            nn.Conv2d(32 +16, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh(),
        )


        if net_opt['cit']:
            self.featureConv = FeatureConv(net_opt=net_opt)

        self.colorConv = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Tanh(),
        )

        if net_opt['guide']:
            self.deconv_for_decoder = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), # output is 64 * 64
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # output is 128 * 128
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # output is 256 * 256
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1, output_padding=0), # output is 256 * 256
                nn.Tanh(),
            )

            
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
            elif hasattr(m, 'weight') and isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _make_encoder_block(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def _make_encoder_block_first(self, inplanes, planes):
        return nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, input,  feature_tensor, c_tag_class):
        # skeleton     512*512*1
        
#        现在的feature_tensor就是 skeleton经过resnext 提取的特征
        out1 = self.conv1(input)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)

        # ==============================
        # it's about color variant tag set
        # temporally, don't think about noise z
        c_tag_tensor = self.Linear(c_tag_class)
        c_tag_tensor = c_tag_tensor.view(-1, 32, self.bottom_h, self.bottom_h)
        c_tag_tensor = self.colorConv(c_tag_tensor)

        c_se_tensor = self.colorFC(c_tag_class)

        # ==============================
        # Convolution Layer for Feature Tensor
        
#         skeleton 的特征
        feature_tensor = self.featureConv(feature_tensor)
        
        mix_cat_tensor = torch.cat([out5,feature_tensor],1)

        mix_out = self.mix(mix_cat_tensor, c_se_tensor)
        
        
        
        concat_tensor = torch.cat([mix_out, c_tag_tensor], 1)
       
        out4_prime = self.deconv1(concat_tensor, c_se_tensor)

        # ==============================
        # Deconv layers

     
        out3_prime = self.deconv2(torch.cat([out4_prime, out4], 1), c_se_tensor)

        out2_prime = self.deconv3(torch.cat([out3_prime, out3], 1), c_se_tensor)

        out1_prime = self.deconv4(torch.cat([out2_prime, out2], 1), c_se_tensor)

        full_output = self.deconv5(torch.cat([out1_prime, out1], 1))

        # ==============================
        # out4_prime should be input of Guide Decoder

        if self.net_opt['guide']:
            decoder_output = self.deconv_for_decoder(out4_prime)
        else:
            decoder_output = full_output
#  不用返回skeleton
        return full_output, decoder_output

class Discriminator(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, input_size=256, cv_class_num=115, iv_class_num=370, person_class_num=10, net_opt=DEFAULT_NET_OPT):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.input_size = input_size
        self.cv_class_num = cv_class_num
        self.iv_class_num = iv_class_num
        self.person_class_num = person_class_num
        self.cardinality = 16

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, 2, 0),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = self._make_block_1(32, 64)
        self.conv3 = self._make_block_1(64, 128)
        self.conv4 = self._make_block_1(128, 256)
        self.conv5 = self._make_block_1(256, 512)
        self.conv6 = self._make_block_3(512, 512)
        self.conv7 = self._make_block_3(512, 512)
        self.conv8 = self._make_block_3(512, 512)
#         自适应池化，每个通道只有一个像素点  512*1*1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.cit_judge = nn.Sequential(
            nn.Linear(512, self.iv_class_num),
            nn.Sigmoid()
        )

        self.cvt_judge = nn.Sequential(
            nn.Linear(512, self.cv_class_num),
            nn.Sigmoid()
        )

        self.adv_judge = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        self.person_judge = nn.Sequential(
            nn.Linear(512, self.person_class_num),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _make_block_1(self, inplanes, planes):
        return nn.Sequential(
            SEResNeXt._make_layer(self, BottleneckX, planes//4, 2, inplanes=inplanes),
            nn.Conv2d(planes, planes, 3, 2, 1),
            nn.LeakyReLU(0.2),
        )

    def _make_block_2(self, inplanes, planes):
        return nn.Sequential(
            SEResNeXt._make_layer(self, BottleneckX, planes//4, 2, inplanes=inplanes),
        )

    def _make_block_3(self, inplanes, planes):
        return nn.Sequential(
            SEResNeXt._make_layer(self, BottleneckX, planes//4, 1, inplanes=inplanes),
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
	
        cit_judge = self.cit_judge(out)
        cvt_judge = self.cvt_judge(out)
        adv_judge = self.adv_judge(out)
        person_judge = self.person_judge(out)
        
        return adv_judge, cit_judge, cvt_judge, person_judge

