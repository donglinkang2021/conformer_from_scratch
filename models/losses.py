# PyTorch
import torch
import torch.nn as nn

class LossCTC(nn.Module):

    def __init__(self):
        super(LossCTC, self).__init__()

        # CTC Loss
        self.loss = nn.CTCLoss(blank=0, reduction="none", zero_infinity=False)

    def forward(self, batch, pred):

        # Unpack Batch
        x, y, x_len, y_len = batch

        if not isinstance(y, torch.Tensor):
            # print("y is not a tensor")
            y = torch.tensor(y)

        # Unpack Predictions
        outputs_pred, f_len, _ = pred

        # Compute Loss
        loss = self.loss(
            log_probs=torch.nn.functional.log_softmax(outputs_pred, dim=-1).transpose(0, 1),
            targets=y,
            input_lengths=f_len,
            target_lengths=y_len
        ).mean()

        return loss

