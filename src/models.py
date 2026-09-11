import torch
import torch.nn as nn
import copy
import timm

class FishClassifier(nn.Module):
    def __init__(self, model_name='convnext_base', num_classes=23, pretrained=True, dropout=0.2):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.num_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Dropout(dropout),
            nn.Linear(self.num_features, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
        
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
            
    def get_param_groups(self, lr_backbone=1e-5, lr_head=1e-4):
        # Split backbone into early and late
        backbone_params = list(self.backbone.named_parameters())
        mid = len(backbone_params) // 2
        
        early_params = [p for n, p in backbone_params[:mid]]
        late_params = [p for n, p in backbone_params[mid:]]
        head_params = list(self.head.parameters())
        
        return [
            {'params': early_params, 'lr': lr_backbone},
            {'params': late_params, 'lr': 3 * lr_backbone},
            {'params': head_params, 'lr': lr_head}
        ]

class ModelEMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.9998):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self, model):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.ema_model.state_dict()
        
    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)


def create_convnext_base(num_classes=23, pretrained=True, dropout=0.2):
    return FishClassifier(
        model_name='convnext_base', 
        num_classes=num_classes, 
        pretrained=pretrained, 
        dropout=dropout
    )

def create_efficientnetv2_m(num_classes=23, pretrained=True, dropout=0.2):
    return FishClassifier(
        model_name='tf_efficientnetv2_m', 
        num_classes=num_classes, 
        pretrained=pretrained, 
        dropout=dropout
    )
