import torch.nn as nn
from torchvision.models import VisionTransformer, vit_b_16, ViT_B_16_Weights


class ViT(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(ViT, self).__init__()

        self.normalize = kwargs.get("normalize", False)
        self.class_num = num_classes
        self.backbone = \
        VisionTransformer(
        image_size=32,
        patch_size=4,
        num_layers=4,
        num_heads=12,
        hidden_dim=192,
        mlp_dim=192*4,
        )
        
        # weights = ViT_B_16_Weights.DEFAULT.get_state_dict()
        # not_matched_module_names = []
        # for a, b in zip(self.backbone.state_dict().items(),weights.items()):
        #     if a[-1].shape != b[-1].shape:
        #         not_matched_module_names.append(a[0])
        # weights = {name:value for name, value in weights.items() if name not in not_matched_module_names}
        # self.backbone.load_state_dict(weights,strict=False)

        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = Deep_Classifier(
            in_features, num_classes, normalize=self.normalize
        )
        self.in_features = in_features

    def forward(self, x, return_feature=False, reverse=False):

        # Reshape and permute the input tensor
        x = self.backbone._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.backbone.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        return self.backbone.heads.head(x, return_feature=return_feature, reverse=reverse)

    def scaling_logits(self, logits, name="cali_scaler"):

        # Expand temperature to match the size of logits
        temperature = (
            getattr(self, name).unsqueeze(1).expand(logits.size(0), logits.size(1))
        )
        temperature = torch.clip(temperature, max=5, min=0.2)

        return logits / (torch.abs(temperature) + 1e-5)

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
