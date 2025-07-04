import torch
import torch.nn as nn
from torch.autograd import Function
from torchvision.models import resnet50


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

# Pre-activation ResNet-50 전체 구조
class ResNet50(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(ResNet50, self).__init__()

        self.normalize = kwargs.get("normalize", False)

        self.backbone = resnet50()
        self.output = Deep_Classifier(
            self.backbone.fc.in_features, num_classes, normalize=self.normalize
        )
        self.in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.class_num = num_classes

    def forward(self, x, return_feature=False, reverse=False):

        feature = self.backbone(x)
        logits = self.output(feature, return_feature=return_feature, reverse=reverse)

        return logits

    def update_batch_stats(self, flag):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.update_batch_stats = flag

    def scaling_logits(self, logits, name="cali_scaler"):

        # Expand temperature to match the size of logits
        temperature = (
            getattr(self, name).unsqueeze(1).expand(logits.size(0), logits.size(1))
        )
        temperature = torch.clip(temperature, max=5, min=0.2)

        return logits / (torch.abs(temperature) + 1e-5)

    def get_only_feat(self, x):

        feature = self.backbone(x)

        if self.normalize:
            feature = nn.functional.normalize(feature)

        return feature


class Deep_Classifier(nn.Module):
    def __init__(self, in_node, out_node, normalize):
        super().__init__()

        if normalize:
            self.linear = nn.Linear(in_node, out_node, bias=False)
        else:
            self.linear = nn.Linear(in_node, out_node)

        self.reversal = GradientReversalLayer()

        self.in_features = in_node
        self.out_features = out_node

        self.normalize = normalize

    def forward(self, feature, reverse=False, return_feature=False):

        if self.normalize:
            feature = nn.functional.normalize(feature)

        if reverse:
            feature = self.reversal(feature)

        logits = self.linear(feature)

        if return_feature:
            return logits, feature
        else:
            return logits
