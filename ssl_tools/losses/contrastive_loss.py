import torch 

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, y_true, y_pred):
        square_pred = y_pred.pow(2)
        margin_square = torch.clamp(self.margin - y_pred, min=0).pow(2)
        loss = torch.mean((1 - y_true) * square_pred + y_true * margin_square)
        return loss