import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
import math

num = 0
act = nn.ReLU(inplace=True)

def save_feature(out, dim, name):
    name = str(num) + name
    filename = "test/tensor_{}_{}.mat".format(dim, name)
    mat = out.cpu().detach().numpy()
    scipy.io.savemat(filename, {"weight":mat})
    print("save feature to {}".format(filename))

def conv_dims(dims):
    if dims == 3:
        f = nn.Conv3d
    elif dims == 2:
        f = nn.Conv2d
    elif dims == 1:
        f = nn.Conv1d
    else:
        raise Exception("dims is not in [1, 2, 3]")

    return f

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, dims=3):
    """3x3 convolution with padding"""
    f = conv_dims(dims)
    return f(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1, dims=3):
    """1x1 convolution"""
    f = conv_dims(dims)
    return f(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def init_weights(m):
    pass

class MMTM(nn.Module):
    def __init__(self, dim_x, dim_y, dim_z, ratio):
        super(MMTM, self).__init__()
        dim = 64
        dim_out = int(3 * dim / ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)
        self.fc_prepare1_a = nn.Linear(dim_x, 16)  
        self.fc_prepare1_m = nn.Linear(dim_x, 16)  
        self.fc_prepare2_a = nn.Linear(dim_y, 16)  
        self.fc_prepare2_m = nn.Linear(dim_y, 16)
        self.fc_x = nn.Linear(dim_out, dim_x)
        self.fc_y = nn.Linear(dim_out, dim_y)
        self.fc_z = nn.Linear(dim_out, dim_z)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # initialize
        with torch.no_grad():
            self.fc_prepare1_a.apply(init_weights)
            self.fc_prepare1_m.apply(init_weights)
            self.fc_prepare2_a.apply(init_weights)
            self.fc_prepare2_m.apply(init_weights)
            self.fc_squeeze.apply(init_weights)
            self.fc_x.apply(init_weights)
            self.fc_y.apply(init_weights)
            self.fc_z.apply(init_weights)

    def forward(self, x, y, z):
        squeeze_average = []
        squeeze_max = []

        for tensor in [x, y, z]:
            tview = tensor.view(tensor.shape[:2] + (-1,))
            squeeze_average.append(torch.mean(tview, dim=-1))
            squeeze_max.append(torch.max(tview, dim=-1)[0])

        squeeze_average[0] = self.fc_prepare1_a(squeeze_average[0])
        squeeze_max[0] = self.fc_prepare1_m(squeeze_max[0])

        squeeze_average[1] = self.fc_prepare2_a(squeeze_average[1])
        squeeze_max[1] = self.fc_prepare2_m(squeeze_max[1])

        squeeze_a = torch.cat(squeeze_average, 1)
        squeeze_m = torch.cat(squeeze_max, 1)

        excitation_a = self.fc_squeeze(squeeze_a)  # fc
        excitation_a = self.relu(excitation_a)  # relu

        x_out_a = self.fc_x(excitation_a) #EA
        y_out_a = self.fc_y(excitation_a) #EB
        z_out_a = self.fc_z(excitation_a) #EC

        excitation_m = self.fc_squeeze(squeeze_m)  # fc
        excitation_m = self.relu(excitation_m)  # relu

        x_out_m = self.fc_x(excitation_m)  # EA
        y_out_m = self.fc_y(excitation_m)  # EB
        z_out_m = self.fc_z(excitation_m)  # EC

        x_out = x_out_a + x_out_m
        y_out = y_out_a + y_out_m
        z_out = z_out_a + z_out_m

        x_out = self.sigmoid(x_out)
        y_out = self.sigmoid(y_out)
        z_out = self.sigmoid(z_out)

        dim_diff = len(x.shape) - len(x_out.shape)
        x_out = x_out.view(x_out.shape + (1,) * dim_diff)

        dim_diff = len(y.shape) - len(y_out.shape)
        y_out = y_out.view(y_out.shape + (1,) * dim_diff)

        dim_diff = len(z.shape) - len(z_out.shape)
        z_out = z_out.view(z_out.shape + (1,) * dim_diff)

        return x * x_out, y * y_out, z * z_out

    def feature_extract(self, x, y, z):
        squeeze_average = []
        squeeze_max = []

        for tensor in [x, y, z]:
            tview = tensor.view(tensor.shape[:2] + (-1,))
            squeeze_average.append(torch.mean(tview, dim=-1))
            squeeze_max.append(torch.max(tview, dim=-1)[0])

        squeeze_average[0] = self.fc_prepare1_a(squeeze_average[0])
        squeeze_max[0] = self.fc_prepare1_m(squeeze_max[0])

        squeeze_average[1] = self.fc_prepare2_a(squeeze_average[1])
        squeeze_max[1] = self.fc_prepare2_m(squeeze_max[1])

        squeeze_a = torch.cat(squeeze_average, 1)
        squeeze_m = torch.cat(squeeze_max, 1)

        excitation_a = self.fc_squeeze(squeeze_a)  
        excitation_a = self.relu(excitation_a)  

        x_out_a = self.fc_x(excitation_a)
        y_out_a = self.fc_y(excitation_a) 
        z_out_a = self.fc_z(excitation_a)

        excitation_m = self.fc_squeeze(squeeze_m) 
        excitation_m = self.relu(excitation_m)  

        x_out_m = self.fc_x(excitation_m) 
        y_out_m = self.fc_y(excitation_m)  
        z_out_m = self.fc_z(excitation_m) 

        x_out = x_out_a + x_out_m
        y_out = y_out_a + y_out_m
        z_out = z_out_a + z_out_m

        x_out = self.sigmoid(x_out)
        y_out = self.sigmoid(y_out)
        z_out = self.sigmoid(z_out)

        dim_diff = len(x.shape) - len(x_out.shape)
        x_out = x_out.view(x_out.shape + (1,) * dim_diff)

        dim_diff = len(y.shape) - len(y_out.shape)
        y_out = y_out.view(y_out.shape + (1,) * dim_diff)

        dim_diff = len(z.shape) - len(z_out.shape)
        z_out = z_out.view(z_out.shape + (1,) * dim_diff)

        return squeeze_a, squeeze_m, excitation_a, excitation_m, x_out, y_out, z_out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, dim=3):
        super(ChannelAttention, self).__init__()
        self.dim = dim
        assert self.dim in [1, 2, 3], "dim [] not in [1, 2, 3]".format(self.dim)
        if dim == 3:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.max_pool = nn.AdaptiveMaxPool3d(1)
        elif dim == 2:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
        else:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.max_pool = nn.AdaptiveMaxPool1d(1)

        conv = conv_dims(dim)
        hidden_planes = in_planes // 16 if in_planes // 16 > 1 else in_planes
        self.fc1   = conv(in_planes, hidden_planes, 1, bias=False)
        self.relu1 = act
        self.fc2   = conv(hidden_planes, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, dim=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        conv = conv_dims(dim)
        self.conv1 = conv(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, dims=3, use_MMTM=None):
        super(ResBlock, self).__init__()
        self.dims = dims
        if norm_layer is None:
            if dims == 3:
                norm_layer = nn.BatchNorm3d
            elif dims == 2:
                norm_layer = nn.BatchNorm2d
            elif dims == 1:
                norm_layer = nn.BatchNorm1d
            else:
                raise ValueError("dims is not in [1, 2, 3]")
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, dims=dims)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dims=dims)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample_x=None, downsample_y=None, downsample_z=None, groups=1, base_width=64, dilation=1, index=1, use_MMTM=True):
        super(BasicBlock, self).__init__()

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.index = index
        self.use_MMTM = use_MMTM
        self.relu = act
        self.conv1_x = conv3x3(inplanes, planes, stride, dims=1)
        self.bn1_x = nn.BatchNorm1d(planes)
        self.conv2_x = conv3x3(planes, planes, dims=1)
        self.bn2_x = nn.BatchNorm1d(planes)

        self.conv1_y = conv3x3(inplanes, planes, stride, dims=2)
        self.bn1_y = nn.BatchNorm2d(planes)
        self.conv2_y = conv3x3(planes, planes, dims=2)
        self.bn2_y = nn.BatchNorm2d(planes)

        self.conv1_z = conv3x3(inplanes, planes, stride, dims=3)
        self.bn1_z = nn.BatchNorm3d(planes)
        self.conv2_z = conv3x3(planes, planes, dims=3)
        self.bn2_z = nn.BatchNorm3d(planes)

        if self.use_MMTM and (self.index == 4 or self.index == 3):
            self.mmtm = MMTM(32, 32, 32, 9)
        else:
            self.ca_x = ChannelAttention(planes, dim=1)
            self.ca_y = ChannelAttention(planes, dim=2)
            self.ca_z = ChannelAttention(planes, dim=3)

        self.sa_x = SpatialAttention(dim=1)
        self.sa_y = SpatialAttention(dim=2)
        self.sa_z = SpatialAttention(dim=3)


        self.downsample_x = downsample_x
        self.downsample_y = downsample_y
        self.downsample_z = downsample_z
        self.stride = stride

    def forward(self, x, y, z):
        identity_x = x
        identity_y = y
        identity_z = z


        out_x = self.conv1_x(x)
        out_x = self.bn1_x(out_x)
        out_x = self.relu(out_x)
        out_x = self.conv2_x(out_x)
        out_x = self.bn2_x(out_x)

        out_y = self.conv1_y(y)
        out_y = self.bn1_y(out_y)
        out_y = self.relu(out_y)
        out_y = self.conv2_y(out_y)
        out_y = self.bn2_y(out_y)

        out_z = self.conv1_z(z)
        out_z = self.bn1_z(out_z)
        out_z = self.relu(out_z)
        out_z = self.conv2_z(out_z)
        out_z = self.bn2_z(out_z)

        global num
        num += 1


        if self.use_MMTM and (self.index == 4 or self.index == 3):
            out_x, out_y, out_z = self.mmtm(out_x, out_y, out_z)
        else:
            out_x_ca = self.ca_x(out_x)
            out_x = out_x_ca * out_x
            out_y_ca = self.ca_y(out_y)
            out_y = out_y_ca * out_y
            out_z_ca = self.ca_z(out_z)
            out_z = out_z_ca * out_z

        # spatial attention
        out_x_sa = self.sa_x(out_x)
        out_x= out_x_sa * out_x
        out_y_sa = self.sa_y(out_y)
        out_y= out_y_sa * out_y
        out_z_sa = self.sa_z(out_z)
        out_z = out_z_sa * out_z


        if self.downsample_x is not None:
            identity_x = self.downsample_x(identity_x)
        if self.downsample_y is not None:
            identity_y = self.downsample_y(y)
        if self.downsample_z is not None:
            identity_z = self.downsample_z(z)

        # resnet identity
        out_x += identity_x
        out_x = self.relu(out_x)
        out_y += identity_y
        out_y = self.relu(out_y)
        out_z += identity_z
        out_z = self.relu(out_z)

        #raise ValueError()
        return out_x, out_y, out_z

class ResNet_3_234_MMTM(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, input_channel=1, use_MMTM=True):
        super(ResNet_3_234_MMTM, self).__init__()

        norm_layerx = nn.BatchNorm1d
        norm_layery = nn.BatchNorm2d
        norm_layerz = nn.BatchNorm3d

        pool_layer_x = nn.MaxPool1d
        ada_pool_layer_x = nn.AdaptiveAvgPool1d
        pool_layer_y = nn.MaxPool2d
        ada_pool_layer_y = nn.AdaptiveAvgPool2d
        pool_layer_z = nn.MaxPool3d
        ada_pool_layer_z = nn.AdaptiveAvgPool3d

        self._norm_layerx = norm_layerx
        self._norm_layery = norm_layery
        self._norm_layerz = norm_layerz

        self.input_channel = input_channel
        self.inplanes = 32 #64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        fx = conv_dims(1)
        fy = conv_dims(2)
        fz = conv_dims(3)

        self.conv1x = fx(self.input_channel, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1x = norm_layerx(self.inplanes)
        self.conv1y = fy(self.input_channel, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1y = norm_layery(self.inplanes)
        self.conv1z = fz(self.input_channel, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1z = norm_layerz(self.inplanes)
        self.relu = act

        # pool_layer 1d 2d 3d따로
        self.maxpool_x = pool_layer_x(kernel_size=3, stride=2, padding=1)
        self.maxpool_y = pool_layer_y(kernel_size=3, stride=2, padding=1)
        self.maxpool_z = pool_layer_z(kernel_size=3, stride=2, padding=1)

        # tdata
        self.index=1

        self.layer1_x = self._make_layer1(ResBlock, 32, layers[0], use_MMTM=use_MMTM)
        self.layer1_y = self._make_layer2(ResBlock, 32, layers[0], use_MMTM=use_MMTM)
        self.layer1_z = self._make_layer3(ResBlock, 32, layers[0], use_MMTM=use_MMTM)

        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, index=2, use_MMTM=use_MMTM)
        self.layer3 = self._make_layer(block, 32, layers[2], stride=2, index=3, use_MMTM=use_MMTM)
        self.layer4 = self._make_layer(block, 32, layers[3], stride=1, index=4, use_MMTM=use_MMTM)

        #ada_pool_layer 1d 2d 따로
        self.avgpool_x = ada_pool_layer_x(1)
        self.avgpool_y = ada_pool_layer_y(1)
        self.avgpool_z = ada_pool_layer_z(1)

        self.fc_x = nn.Linear(32 * block.expansion, num_classes)
        self.fc_y = nn.Linear(32 * block.expansion, num_classes)
        self.fc_z = nn.Linear(32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, index=1, use_MMTM=True, dilate=False):
        norm_layerx = self._norm_layerx
        norm_layery = self._norm_layery
        norm_layerz = self._norm_layerz
        downsample_x = None
        downsample_y = None
        downsample_z = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample_x = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride, dims=1),
                norm_layerx(planes * block.expansion),
            )
            downsample_y = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride, dims=2),
                norm_layery(planes * block.expansion),
            )
            downsample_z = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride, dims=3),
                norm_layerz(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample_x, downsample_y, downsample_z, self.groups,
                            self.base_width, previous_dilation, index=index, use_MMTM=use_MMTM))

        return layers[0]

    def _make_layer1(self, block, planes, blocks, stride=1, dilate=False, use_MMTM=True):
        norm_layer = self._norm_layerx
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride, dims=1),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, dims=1, use_MMTM=use_MMTM))
        return nn.Sequential(*layers)

    def _make_layer2(self, block, planes, blocks, stride=1, dilate=False, use_MMTM=True):
        norm_layer = self._norm_layery
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride, dims=2),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, dims=2, use_MMTM=use_MMTM))

        return nn.Sequential(*layers)

    def _make_layer3(self, block, planes, blocks, stride=1, dilate=False, use_MMTM=True):
        norm_layer = self._norm_layerz
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride, dims=3),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, dims=3, use_MMTM=use_MMTM))
        return nn.Sequential(*layers)


    def _forward_impl(self, x, y, z):
        # x(ts), y(wt)
        x = self.conv1x(x)
        x = self.bn1x(x)
        x = self.relu(x)
        x = self.maxpool_x(x)

        y = self.conv1y(y)
        y = self.bn1y(y)
        y = self.relu(y)
        y = self.maxpool_y(y)

        z = self.conv1z(z)
        z = self.bn1z(z)
        z = self.relu(z)
        z = self.maxpool_z(z)

        x = self.layer1_x(x)
        y = self.layer1_y(y)
        z = self.layer1_z(z)

        x, y, z = self.layer2(x, y, z)

        x, y, z = self.layer3(x, y, z)

        x, y, z = self.layer4(x, y, z)

        x = self.avgpool_x(x)
        x = torch.flatten(x, 1)
        x = self.fc_x(x)

        y = self.avgpool_y(y)
        y = torch.flatten(y, 1)
        y = self.fc_y(y)

        z = self.avgpool_z(z)
        z = torch.flatten(z, 1)
        z = self.fc_z(z)

        return x, y, z

    def forward(self, x, y, z):
        return self._forward_impl(x, y, z)

    def extract_feature(self, x, y, z):
        x = self.conv1x(x)
        x = self.bn1x(x)
        x = self.relu(x)
        x = self.maxpool_x(x)

        y = self.conv1y(y)
        y = self.bn1y(y)
        y = self.relu(y)
        y = self.maxpool_y(y)

        z = self.conv1z(z)
        z = self.bn1z(z)
        z = self.relu(z)
        z = self.maxpool_z(z)

        x = self.layer1_x(x)
        y = self.layer1_y(y)
        z = self.layer1_z(z)

        x, y, z = self.layer2(x, y, z)

        x, y, z = self.layer3(x, y, z)

        x, y, z = self.layer4(x, y, z)

        return x, y, z


class CWConv(nn.Module): # set frequency as param.
    def __init__(self, first_freq=1, last_freq=100, filter_n=100, kernel_size=7, in_channels=1):
        # print(f"kernel size : {kernel_size}")
        super(CWConv, self).__init__()
        if in_channels != 1:
            msg = "CWConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)
        self.first_freq = first_freq
        self.last_freq = last_freq
        self.kernel_size = kernel_size
        self.filter_n = filter_n
        self.omega = 5.15
        self.a_ = nn.parameter.Parameter(torch.tensor([float(x/100) for x in range(first_freq, last_freq+1)]).view(-1, 1)) # frequency component vector
        self.b_ = torch.tensor(self.omega)
        # return the frequency components
    def get_freq_info(self):
        return torch.clamp(torch.tensor(list(map(lambda x: float(x * 100.0), self.a_))), min=1e-7)
        # return the omega
    def get_omega_info(self):
        return torch.tensor(self.b_)
    def forward(self, waveforms):
        device = waveforms.device
        M = self.kernel_size
        # x -> -74.5 ~ 74.5 with a interval of 1. total 150
        x = (torch.arange(0, M) - (M - 1.0) / 2).to(device)
        s = (2.5 * self.b_) / (torch.clamp(self.a_, min=1e-7) * 2 * math.pi) # represent frequency componenets as scale params
        x = x / s
        wavelet = (torch.cos(self.b_ * x) * torch.exp(-0.5 * x ** 2) * math.pi ** (-0.25))
        output = (torch.sqrt(1 / s) * wavelet)
        Morlet_filter = output
        self.filters = (Morlet_filter).view(self.filter_n, 1, self.kernel_size)
        out = F.conv1d(waveforms, self.filters, stride=1, padding=(self.kernel_size-1)//2, dilation=1, bias=None, groups=1)
        return out
    


def _resnet3_234_mmtm(block, layers, num_class=1000, input_channel=1, use_MMTM=True, **kwargs):
    model = ResNet_3_234_MMTM(block, layers, num_class, input_channel=input_channel, use_MMTM=use_MMTM, **kwargs)
    return model


def resnet_mmtm(num_class=1000, input_channel=1, use_MMTM=True, **kwargs):
    return _resnet3_234_mmtm(BasicBlock, [1, 1, 1, 1], num_class=num_class, input_channel=input_channel, use_MMTM=use_MMTM, **kwargs)


def get_weight(load_model_name, model_weights, print_err=False):
    param = torch.load(load_model_name)

    dic = {}
    missing_weight = []
    weight_keys = model_weights.keys()
    for k in weight_keys:
        if k in param:
            dic[k] = param[k].clone()
        else:
            missing_weight.append(k)

    if print_err:
        missing_param = list(set(param.keys()) - set(weight_keys))

        if missing_weight:
            print(f"Warning : loaded parameter has no weight named {missing_weight} ignoring...")

        if missing_param:
            print(f"Warning : model has no weight named {missing_param} ignoring...")

    return dic


class ConHead(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(ConHead, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return F.normalize(self.linear(x), p=2.0, dim=1)