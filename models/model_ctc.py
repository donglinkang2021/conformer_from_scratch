# PyTorch
import torch
import torch.nn as nn

# Base Model
from models.model import Model

# Encoders
from models.encoders import (
    ConformerEncoder
)

# Losses
from models.losses import (
    LossCTC
)

class ModelCTC(Model):

    def __init__(self, encoder_params, tokenizer_params, training_params, name):
        super(ModelCTC, self).__init__(tokenizer_params, training_params, name)

        # Encoder
        self.encoder = ConformerEncoder(encoder_params)

        # FC Layer
        self.fc = nn.Linear(
            encoder_params["dim_model"], 
            tokenizer_params["vocab_size"]
        )

        # Criterion
        self.criterion = LossCTC()

        # Compile
        self.compile(training_params)

    def forward(self, batch):

        # Unpack Batch
        x, _, x_len, _ = batch

        # Forward Encoder (B, Taud) -> (B, T, Denc)
        logits, logits_len, attentions = self.encoder(x, x_len)

        # FC Layer (B, T, Denc) -> (B, T, V)
        logits = self.fc(logits)

        return logits, logits_len, attentions

    def gready_search_decoding(self, x, x_len):

        # Forward Encoder (B, Taud) -> (B, T, Denc)
        logits, logits_len = self.encoder(x, x_len)[:2]

        # FC Layer (B, T, Denc) -> (B, T, V)
        logits = self.fc(logits)

        # Softmax -> Log > Argmax -> (B, T)
        preds = logits.log_softmax(dim=-1).argmax(dim=-1)

        # Batch Pred List
        batch_pred_list = []

        # Batch loop
        for b in range(logits.size(0)):

            # Blank
            blank = False

            # Pred List
            pred_list = []

            # Decoding Loop
            for t in range(logits_len[b]):

                # Blank Prediction
                if preds[b, t] == 0:
                    blank = True
                    continue

                # First Prediction
                if len(pred_list) == 0:
                    pred_list.append(preds[b, t].item())

                # New Prediction
                elif pred_list[-1] != preds[b, t] or blank:
                    pred_list.append(preds[b, t].item())
                
                # Update Blank
                blank = False

            # Append Sequence
            batch_pred_list.append(pred_list)

        # Decode Sequences
        # TODO: 改这里的解码方式
        decoded_pred_list = []
        for item in batch_pred_list:
            decoded_pred_list.append(" ".join(self.tokenizer.to_tokens(item)))

        # Decode Sequences
        return decoded_pred_list
