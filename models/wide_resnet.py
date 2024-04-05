import torch
import torch.nn as nn

from torch.autograd import Function

class GradientReversalFunction(Function):
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        output = grad_output.neg()
        return output

class GradientReversalLayer(nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def forward(self, x):
        return GradientReversalFunction.apply(x)
        
def conv3x3(i_c, o_c, stride=1):
    return nn.Conv2d(i_c, o_c, 3, stride, 1, bias=False)

class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, channels, eps=1e-3, momentum=1e-3):
        super().__init__(channels)
        self.update_batch_stats = True
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        if self.update_batch_stats:
            return super().forward(x)
        else:
            return nn.functional.batch_norm(
                x, None, None, self.weight, self.bias, True, self.momentum, self.eps
            )

def relu():
    return nn.LeakyReLU(0.1)

class residual(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, activate_before_residual=False):
        super().__init__()
        layer = []
        if activate_before_residual:
            self.pre_act = nn.Sequential(
                BatchNorm2d(input_channels),
                relu()
            )
        else:
            self.pre_act = nn.Identity()
            layer.append(BatchNorm2d(input_channels))
            layer.append(relu())
        layer.append(conv3x3(input_channels, output_channels, stride))
        layer.append(BatchNorm2d(output_channels))
        layer.append(relu())
        layer.append(conv3x3(output_channels, output_channels))

        if stride >= 2 or input_channels != output_channels:
            self.identity = nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)
        else:
            self.identity = nn.Identity()

        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        x = self.pre_act(x)
        return self.identity(x) + self.layer(x)

class WRN(nn.Module):
    """ WRN28-width with leaky relu (negative slope is 0.1)"""
    def __init__(self, width, num_classes, **kwargs):
        super().__init__()

        self.init_conv = conv3x3(3, 16)

        filters = [16, 16*width, 32*width, 64*width]

        unit1 = [residual(filters[0], filters[1], activate_before_residual=True)] + \
            [residual(filters[1], filters[1]) for _ in range(1, 4)]
        self.unit1 = nn.Sequential(*unit1)

        unit2 = [residual(filters[1], filters[2], 2)] + \
            [residual(filters[2], filters[2]) for _ in range(1, 4)]
        self.unit2 = nn.Sequential(*unit2)

        unit3 = [residual(filters[2], filters[3], 2)] + \
            [residual(filters[3], filters[3]) for _ in range(1, 4)]
        self.unit3 = nn.Sequential(*unit3)

        self.unit4 = nn.Sequential(*[BatchNorm2d(filters[3]), relu(), nn.AdaptiveAvgPool2d(1)])

        self.normalize = kwargs.get('normalize',False)

        self.output = Deep_Classifier(filters[3], num_classes, normalize=self.normalize)

        self.class_num = num_classes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")    
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if not self.normalize:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_feature=False, reverse=False):

        x = self.init_conv(x)
        x = self.unit1(x)
        x = self.unit2(x)
        x = self.unit3(x)
        x = self.unit4(x)

        return self.output(x.squeeze(-1).squeeze(-1), return_feature=return_feature, reverse=reverse)

    def update_batch_stats(self, flag):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.update_batch_stats = flag

    def scaling_logits(self, logits, name='cali_scaler'):

        # Expand temperature to match the size of logits
        temperature = getattr(self,name).unsqueeze(1).expand(logits.size(0), logits.size(1))
        
        if name!='cali_scaler':
            temperature = torch.clip(temperature,max=3)
        
        return logits / (torch.abs(temperature)+1e-5)

    def get_only_feat(self, x):

        x = self.init_conv(x)
        x = self.unit1(x)
        x = self.unit2(x)
        x = self.unit3(x)
        x = self.unit4(x)
        
        feature = x.squeeze(-1).squeeze(-1)

        if self.normalize:
            feature = nn.functional.normalize(feature)

        return feature
    
class Deep_Classifier(nn.Module):
    def __init__(self, in_node, out_node, normalize):
        super().__init__()
        
        if normalize:
            self.linear = nn.Linear(in_node,out_node, bias=False)
        else:
            self.linear = nn.Linear(in_node,out_node)
            
        self.reversal = GradientReversalLayer()
        
        self.in_features = in_node
        self.out_features = out_node
        
        self.normalize = normalize
        
    def forward(self,feature,reverse=False, return_feature=False):
        
        if self.normalize:
            feature = nn.functional.normalize(feature)
        
        if reverse:
            feature = self.reversal(feature)
        
        logits = self.linear(feature)
        
        if return_feature:
            return logits, feature
        else:
            return logits
        
# Sub-Network
class LULClassifier(nn.Module):
    def __init__(self, feature, class_num, size, lambda_weight : float = 1e-5) -> None:
        super().__init__()
        
        assert size>=2
        
        self.lambda_weight = lambda_weight
        
        modules = []
        for _ in range(size-1):
            modules.append(nn.Linear(feature,feature))
            modules.append(nn.LayerNorm(feature))
            modules.append(nn.LeakyReLU(0.1))
        modules.append(nn.Linear(feature,class_num, bias=False))
        
        self.reversal = GradientReversalLayer()
        self.mlp = nn.Sequential(*modules)
        
    def forward(self,x,reversal=False):

        if reversal:
            x = self.reversal(x)

        return self.mlp(x)
    
    def l2_norm_loss(self):

        sum_of_squares = 0
        for param in self.mlp.parameters():
            sum_of_squares += torch.sum(torch.pow(param, 2))

        weights_reg = self.lambda_weight * sum_of_squares

        return weights_reg