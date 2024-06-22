import torch
import torch.nn as nn
import torch.nn.functional as F


# class FocalLoss(nn.Module):
#     "Focal loss implemented using F.cross_entropy"
#     def __init__(self, gamma: float = 2.0, weight=None, reduction: str = 'mean') -> None:
#         super().__init__()
#         self.gamma = gamma
#         self.weight = weight
#         self.reduction = reduction


#     def forward(self, inp: torch.Tensor, targ: torch.Tensor):
#         ce_loss = F.cross_entropy(inp, targ, weight=self.weight, reduction="none")
#         p_t = torch.exp(-ce_loss)
#         loss = (1 - p_t)**self.gamma * ce_loss
#         if self.reduction == "mean":
#             loss = loss.mean()
#         elif self.reduction == "sum":
#             loss = loss.sum()
#         return loss

class FocalLoss(nn.Module):
    """
    Simple pytorch implementation of focal loss introduced
    by *Lin et al.* (https://arxiv.org/pdf/1708.02002.pdf).
    """
    
    def __init__(self, gamma=2, alpha=0.5, weight=None, ignore_index=None):
        """Initialize the focal loss.

        Args:
            gamma (int, optional): exponent of the modulating factor (1 - p_t) to balance easy vs hard examples. Defaults to 2.
            alpha (float, optional): weighting factor in range (0,1) to balance positive vs negative examples. Defaults to 0.5.
            weight (Tensor, optional): a manual rescaling weight given to each class. Defaults to None.
            ignore_index (int, optional): target value that is ignored and does not contribute to the input gradient. Defaults to 255.
        """        
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=-1)

    def forward(self, preds, labels):
        """Foward function use during trainning. 
       
        Args:
            preds (Tensor): the output from model.
            labels (Tensor): ground truth.

        Returns:
            Tensor: focal loss score.
        """        
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        if self.alpha is not None:
            logpt *= self.alpha
        loss = -((1 - pt) ** self.gamma) * logpt

        return loss
  
# class DynamicFocalLoss(torch.nn.Module):
#     def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
#         super(DynamicFocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha  # Tensor of alpha values for each class
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         # For multiclass, inputs are assumed to be raw logits
#         ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
#         p_t = torch.exp(-ce_loss)
#         if self.alpha is not None:
#             # Assuming targets are class indices for multiclass
#             at = self.alpha[targets]
#         else:
#             at = 1.0
#         loss = at * (1 - p_t) ** self.gamma * ce_loss
#         if self.reduction == "mean":
#             return loss.mean()
#         elif self.reduction == "sum":
#             return loss.sum()
#         return loss

# class BinaryFocalLoss(nn.Module):
#     def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
#         super(BinaryFocalLoss, self).__init__()
#         self.gamma = gamma
#         # Ensure alpha is a tensor of size 2, even for binary classification
#         if alpha is not None and torch.is_tensor(alpha):
#             self.alpha = alpha
#         else:
#             # If alpha is not specified or not a tensor, set a default balanced value
#             self.alpha = torch.tensor([0.5, 0.5])
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         # Apply sigmoid to logits to get probabilities
#         probs = torch.sigmoid(inputs)
#         # Calculate the binary cross-entropy loss without reduction
#         bce_loss = F.binary_cross_entropy(probs, targets, reduction='none')
        
#         targets = targets.type(torch.long)
#         # Select the correct alpha value for each pixel based on its target class
#         alpha_factor = torch.where(torch.eq(targets, 1), self.alpha[1], self.alpha[0])
        
#         # Calculate the modulating factor
#         p_t = torch.where(torch.eq(targets, 1), probs, 1 - probs)
#         modulating_factor = (1.0 - p_t) ** self.gamma
        
#         # Apply the focal loss formula
#         focal_loss = alpha_factor * modulating_factor * bce_loss
        
#         # Apply reduction
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss


# class MulticlassFocalLoss(nn.Module):
#     def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
#         super(MulticlassFocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha  # Tensor of alpha values for each class
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         # Ensure alpha is on the same device as inputs
#         if self.alpha is not None:
#             alpha = self.alpha.to(inputs.device)
#         else:
#             alpha = torch.ones(inputs.size(1), device=inputs.device)  # Assuming alpha is 1 for all classes by default
        
#         # Convert alpha to the correct format
#         alpha = alpha[None, :, None, None]
        
#         # Compute softmax over the classes axis
#         inputs_softmax = F.softmax(inputs, dim=1)
#         inputs_log_softmax = F.log_softmax(inputs, dim=1)
        
#         # Gather the probabilities and log probabilities for target classes
#         targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2)
#         probs = torch.sum(targets_one_hot * inputs_softmax, dim=1)
#         log_probs = torch.sum(targets_one_hot * inputs_log_softmax, dim=1)
        
#         # Calculate the focal loss component
#         focal_loss = -alpha * ((1 - probs) ** self.gamma) * log_probs
        
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         return focal_loss


if __name__ == "__main__":
    print()