import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        num_classes = y_pred.shape[1]
        y_true_onehot = F.one_hot(y_true, num_classes).permute(0, 3, 1, 2).float()
        y_pred_softmax = F.softmax(y_pred, dim=1)
        
        intersection = (y_pred_softmax * y_true_onehot).sum(dim=(2, 3))
        union = y_pred_softmax.sum(dim=(2, 3)) + y_true_onehot.sum(dim=(2, 3))
        
        dice_per_class = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_per_class.mean()  # Moyenne sur les classes
    

class CEDiceLoss(nn.Module):
    """
    Combined Cross-Entropy and Dice Loss.
    CE handles class imbalance while Dice enhances segmentation performance.
    
    Args:
        weight (Tensor, optional): Manual rescaling weight for each class.
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'
        smooth (float): Smoothing factor for Dice calculation
    """
    def __init__(self, weight=None, reduction='mean', smooth=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, input, target):
        # Input check
        if input.size() != target.size():
            raise ValueError(f"Input size ({input.size()}) must match target size ({target.size()})")
            
        # Cross-Entropy Loss
        ce_loss = self.ce(input, target)
        
        # Convert to probabilities (softmax for multi-class)
        pred = torch.softmax(input, dim=1)
        
        # Prepare target for Dice (one-hot encoding)
        if target.dim() == 3:  # If target is class indices
            target = torch.nn.functional.one_hot(target, num_classes=input.shape[1]).permute(0, 3, 1, 2).float()
        
        # Calculate Dice for each class
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_coef = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_coef
        
        # Combine losses
        if self.reduction == 'mean':
            total_loss = ce_loss + dice_loss.mean()
        elif self.reduction == 'sum':
            total_loss = ce_loss + dice_loss.sum()
        else:  # 'none'
            total_loss = ce_loss + dice_loss
            
        return total_loss
        
class CEDiceLossWithDS(nn.Module):
    """Version adaptée pour Deep Supervision"""
    def __init__(self, weight=None, reduction='mean'):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        self.smooth = 1.0

    def forward(self, outputs, target):
        final_out, ds1_out, ds2_out = outputs
        
        # Loss pour la sortie finale
        loss_final = self._compute_loss(final_out, target)
        
        # Adapte les targets aux résolutions intermédiaires
        target_ds1 = F.interpolate(target.unsqueeze(1).float(), scale_factor=0.25, mode='nearest').squeeze(1).long()
        target_ds2 = F.interpolate(target.unsqueeze(1).float(), scale_factor=0.5, mode='nearest').squeeze(1).long()
        
        # Loss pour les sorties intermédiaires
        loss_ds1 = self._compute_loss(ds1_out, target_ds1)
        loss_ds2 = self._compute_loss(ds2_out, target_ds2)
        
        # Combinaison pondérée
        return loss_final + 0.5 * loss_ds1 + 0.3 * loss_ds2

    def _compute_loss(self, pred, target):
        # Cross-Entropy
        ce_loss = self.ce(pred, target)
        
        # Dice
        pred_softmax = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (pred_softmax * target_onehot).sum(dim=(2, 3))
        union = pred_softmax.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
        dice_loss = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
        
        return ce_loss + dice_loss.mean()