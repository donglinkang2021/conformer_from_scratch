# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Sentencepiece
import sentencepiece as spm

# Schedulers
from models.schedules import *

# Other
from tqdm import tqdm
import jiwer
import os
import time
import re

from paddlespeech.s2t.frontend.featurizer import TextFeaturizer

def add_space_after_chinese(text):
    pattern = r'([\u4e00-\u9fff])'
    replaced_text = re.sub(pattern, r'\1 ', text)
    return replaced_text

class Model(nn.Module):

    def __init__(self, tokenizer_params, training_params, name):
        super(Model, self).__init__()

        # Tokenizer
        self.tokenizer = TextFeaturizer(unit_type='char',
                                        vocab='utils/paddle/vocab.txt')

        # Training Params
        self.encoder_frozen_steps = training_params.get("encoder_frozen_steps", None)
        self.vn_start_step = training_params.get("vn_start_step", None)

        # Model Name
        self.name = name

    def compile(self, training_params):

        # Optimizers
        # Adam
        self.optimizer = optim.Adam(
            params=self.parameters(), 
            lr=0, 
            betas=(training_params["beta1"], training_params["beta2"]), 
            eps=training_params["eps"], 
            weight_decay=training_params["weight_decay"]
        )


        # LR Schedulers
        # Transformer LR
        self.scheduler = transformer_learning_rate_scheduler(
            optimizer=self.optimizer, 
            dim_model=training_params["schedule_dim"], 
            warmup_steps=training_params["warmup_steps"], 
            K=training_params["K"])

        # Init LR
        self.scheduler.step()

    def num_params(self):
        return sum([p.numel() for p in self.parameters()])

    def summary(self):

        print(self.name)
        print("Model Parameters :", self.num_params())


    def fit(self, dataset_train, epochs, dataset_val=None, val_steps=None, verbose_val=False, initial_epoch=0, callback_path=None, steps_per_epoch=None, mixed_precision=False, accumulated_steps=1, saving_period=1, val_period=1):

        # Model Device
        device = next(self.parameters()).device

        # Mixed Precision Gradient Scaler
        scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

        # Init Training
        acc_step = 0
        self.optimizer.zero_grad()

        # Callbacks
        if callback_path is not None:

             # Create Callbacks
            if not os.path.isdir(callback_path):
                os.makedirs(callback_path)

            # Create Writer
            writer = SummaryWriter(callback_path + "logs")

        else:

            writer = None

        
        # Try Catch
        try:

            # Training Loop
            for epoch in range(initial_epoch, epochs):

                # Epoch Init
                print("Epoch {}/{}".format(epoch + 1, epochs))
                epoch_iterator = tqdm(dataset_train, total=steps_per_epoch * accumulated_steps if steps_per_epoch else None)
                epoch_loss = 0.0

                # Training Mode
                self.train()

                # Epoch training
                for step, batch in enumerate(epoch_iterator):

                    # Load batch to model device
                    batch = [elt.to(device) for elt in batch]

                    # Encoder Frozen Steps
                    if self.encoder_frozen_steps:
                        if self.scheduler.model_step > self.encoder_frozen_steps:
                            self.encoder.requires_grad_(True)
                        else:
                            self.encoder.requires_grad_(False)

                    # Automatic Mixed Precision Casting (model prediction + loss computing)
                    with torch.cuda.amp.autocast(enabled=mixed_precision):
                        pred = self.forward(batch)
                        loss_mini = self.criterion(batch, pred)
                        loss = loss_mini / accumulated_steps

                    # Accumulate gradients
                    scaler.scale(loss).backward()

                    # Update Epoch Variables
                    acc_step += 1
                    epoch_loss += loss_mini.detach()

                    # Continue Accumulating
                    if acc_step < accumulated_steps:
                        continue

                    # Update Parameters, Zero Gradients and Update Learning Rate
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    acc_step = 0


                    # Step Print
                    epoch_iterator.set_description("model step: {} - mean loss {:.4f} - batch loss: {:.4f} - learning rate: {:.6f}".format(
                            self.scheduler.model_step, 
                            epoch_loss / (step + 1), 
                            loss_mini, 
                            self.optimizer.param_groups[0]['lr']
                        )
                    )

                    # Logs Step
                    if writer is not None and (step + 1) % 10 == 0:
                        writer.add_scalar(
                            'Training/Loss', 
                            loss_mini, 
                            self.scheduler.model_step
                        )
                        writer.add_scalar(
                            'Training/LearningRate',  
                            self.optimizer.param_groups[0]['lr'], 
                            self.scheduler.model_step
                        )

                    # Step per Epoch
                    if steps_per_epoch is not None:
                        if step + 1 >= steps_per_epoch * accumulated_steps:
                            break

                # Reduce Epoch Loss among devices

                # Logs Epoch
                if writer is not None:
                    writer.add_scalar(
                        'Training/MeanLoss', 
                        epoch_loss / (steps_per_epoch * accumulated_steps if steps_per_epoch is not None else dataset_train.__len__()),  
                        epoch + 1
                    )

                # Validation
                if (epoch + 1) % val_period == 0:

                    # Validation Dataset
                    if dataset_val:

                        # Multiple Validation Datasets
                        if isinstance(dataset_val, dict):

                            for dataset_name, dataset in dataset_val.items():

                                # Evaluate
                                wer, truths, preds, val_loss = self.evaluate(dataset, val_steps, verbose_val, eval_loss=True)

                                # Print wer
                                print("{} wer : {:.2f}% - loss : {:.4f}".format(dataset_name, 100 * wer, val_loss))

                                # Logs Validation
                                if writer is not None:
                                    writer.add_scalar('Validation/WER/{}'.format(dataset_name), 100 * wer, epoch + 1)
                                    writer.add_scalar('Validation/MeanLoss/{}'.format(dataset_name), val_loss, epoch + 1)
                                    writer.add_text('Validation/Predictions/{}'.format(dataset_name), "GroundTruth : " + truths[0] + " / Prediction : " + preds[0], epoch + 1)

                        else:

                            # Evaluate
                            wer, truths, preds, val_loss = self.evaluate(dataset_val, val_steps, verbose_val, eval_loss=True)

                            # Print wer
                            print("Val wer : {:.2f}% - Val loss : {:.4f}".format(100 * wer, val_loss))

                            # Logs Validation
                            if writer is not None:
                                writer.add_scalar('Validation/WER', 100 * wer, epoch + 1)
                                writer.add_scalar('Validation/MeanLoss', val_loss, epoch + 1)
                                writer.add_text('Validation/Predictions', "GroundTruth : " + truths[0] + " / Prediction : " + preds[0], epoch + 1)

                # Saving Checkpoint
                if (epoch + 1) % saving_period == 0:
                    if callback_path is not None:
                        self.save(callback_path + "checkpoints_" + str(epoch + 1) + ".ckpt")

        # Exception Handler
        except Exception as e:

            if writer is not None:
                writer.add_text('Exceptions', str(e))

            raise e

    def save(self, path, save_optimizer=True):
        
        # Save Model Checkpoint
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if save_optimizer else None,
            "model_step": self.scheduler.model_step,
            "tokenizer": self.tokenizer,
        }, path)

        # Print Model state
        print("model saved at step {} / lr {:.6f}".format(self.scheduler.model_step, self.optimizer.param_groups[0]['lr']))

    def load(self, path):

        # Load Model Checkpoint
        checkpoint = torch.load(path, map_location=next(self.parameters()).device)

        # Model State Dict
        self.load_state_dict({key:value for key, value in checkpoint["model_state_dict"].items()})

        # Model Step
        self.scheduler.model_step = checkpoint["model_step"]

        # Optimizer State Dict
        if checkpoint["optimizer_state_dict"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Tokenizer
        self.tokenizer = checkpoint["tokenizer"]

        # Print Model state
        print("model loaded at step {} / lr {:.6f}".format(self.scheduler.model_step, self.optimizer.param_groups[0]['lr']))

    def evaluate(self, dataset_eval, eval_steps=None, verbose=False, beam_size=1, eval_loss=True):

        # Evaluzation Mode
        self.eval()

        # Model Device
        device = next(self.parameters()).device

        # Groundtruth / Prediction string lists
        speech_true = []
        speech_pred = []

        # Total wer / loss
        total_wer = 0.0
        total_loss = 0.0

        # tqdm Iterator
        eval_iterator = tqdm(dataset_eval, total=eval_steps)

        # Evaluation Loop
        for step, batch in enumerate(eval_iterator):

            batch = [elt.to(device) for elt in batch]

            # Sequence Prediction
            with torch.no_grad():

                if beam_size > 1:
                    outputs_pred = self.beam_search_decoding(batch[0], batch[2], beam_size)
                else:
                    outputs_pred = self.gready_search_decoding(batch[0], batch[2])

            # Sequence Truth
            batch[1] = batch[1].cpu().numpy().tolist()
            outputs_true = []
            for item in batch[1]:
                outputs_true.append(add_space_after_chinese(self.tokenizer.defeaturize(item)))

            # Compute Batch wer and Update total wer
            batch_wer = jiwer.wer(outputs_true, outputs_pred)
            total_wer += batch_wer

            # Update String lists
            speech_true += outputs_true
            speech_pred += outputs_pred

            # Prediction Verbose
            if verbose:
                print("Groundtruths :\n", outputs_true)
                print("Predictions :\n", outputs_pred)

            # Eval Loss
            if eval_loss:
                with torch.no_grad():
                    pred = self.forward(batch)
                    batch_loss = self.criterion(batch, pred)
                    total_loss += batch_loss

            # Step print
            if eval_loss:
                eval_iterator.set_description("mean batch wer {:.2f}% - batch wer: {:.2f}% - mean loss {:.4f} - batch loss: {:.4f}".format(100 * total_wer / (step + 1), 100 * batch_wer, total_loss / (step + 1), batch_loss))
            else:
                eval_iterator.set_description("mean batch wer {:.2f}% - batch wer: {:.2f}%".format(100 * total_wer / (step + 1), 100 * batch_wer))

            # Evaluation Steps
            if eval_steps:
                if step + 1 >= eval_steps:
                    break

        # Compute wer
        if total_wer / (eval_steps if eval_steps is not None else dataset_eval.__len__()) > 1:
            wer = 1
        else:
            wer = jiwer.wer(speech_true, speech_pred, standardize=True)

        # Compute loss
        if eval_loss:
            loss = total_loss / (eval_steps if eval_steps is not None else dataset_eval.__len__())

        # Return word error rate, groundtruths and predictions
        return wer, speech_true, speech_pred, loss if eval_loss else None

    def swa(self, dataset, callback_path, start_epoch, end_epoch, epochs_list=None, update_steps=None, swa_type="equal", swa_decay=0.9):

        # Model device
        device = next(self.parameters()).device

        # Create SWA Model
        if swa_type == "equal":
            swa_model = torch.optim.swa_utils.AveragedModel(self)
        elif swa_type == "exp":
            swa_model = torch.optim.swa_utils.AveragedModel(self, avg_fn=lambda averaged_model_parameter, model_parameter, num_averaged: (1 - swa_decay) * averaged_model_parameter + swa_decay * model_parameter)

        if epochs_list:
            print("Stochastic Weight Averaging on checkpoints : {}".format(epochs_list))
        else:
            print("Stochastic Weight Averaging on checkpoints : {}-{}".format(start_epoch, end_epoch))

        # Update SWA Model Params
        if epochs_list:

            for epoch in epochs_list:

                # Load Model Checkpoint
                self.load(callback_path + "checkpoints_" + str(epoch) + ".ckpt")

                # Update SWA Model
                swa_model.update_parameters(self)

        else:

            for epoch in range(int(start_epoch), int(end_epoch) + 1):

                # Load Model Checkpoint
                self.load(callback_path + "checkpoints_" + str(epoch) + ".ckpt")

                # Update SWA Model
                swa_model.update_parameters(self)

        # Load SWA Model Params
        self.load_state_dict({key[7:]:value for key, value in swa_model.state_dict().items() if key != "n_averaged"})

        print("Updating Batch Normalization Statistics")

        # Init
        self.train()
        dataset_iterator = tqdm(dataset, total=update_steps)

        # Update Batch Normalization Statistics
        for step, batch in enumerate(dataset_iterator):

            # Load batch to model device
            batch = [elt.to(device) for elt in batch]

            # Forward Encoder
            with torch.cuda.amp.autocast(enabled=True):
                with torch.no_grad():
                    self.encoder.forward(batch[0], batch[2])

            # update_steps
            if update_steps is not None:
                if step + 1 == update_steps:
                    break

        # Save Model
        if epochs_list:
            self.save(callback_path + "checkpoints_swa-" + swa_type + "-" + "list" + "-" + epochs_list[0] + "-"  + epochs_list[-1] + ".ckpt", save_optimizer=False)
        else:
            self.save(callback_path + "checkpoints_swa-" + swa_type + "-" + start_epoch + "-"  + end_epoch + ".ckpt", save_optimizer=False)


    def eval_time(self, dataset_eval, eval_steps=None, beam_size=1, rnnt_max_consec_dec_steps=None, profiler=False):

        def decode():

            # Start Timer
            start = time.time()

            # Evaluation Loop
            for step, batch in enumerate(eval_iterator):

                batch = [elt.to(device) for elt in batch]

                # Sequence Prediction
                with torch.no_grad():

                    if beam_size > 1:
                        outputs_pred = self.beam_search_decoding(batch[0], batch[2], beam_size)
                    else:
                        if rnnt_max_consec_dec_steps is not None:
                            outputs_pred = self.gready_search_decoding(batch[0], batch[2], rnnt_max_consec_dec_steps)
                        else:
                            outputs_pred = self.gready_search_decoding(batch[0], batch[2])

                # Evaluation Steps
                if eval_steps:
                    if step + 1 >= eval_steps:
                        break
            # Stop Timer
            return time.time() - start

        # Model Device
        device = next(self.parameters()).device

        # Evaluzation Mode
        self.eval()

        # tqdm Iterator
        eval_iterator = tqdm(dataset_eval, total=eval_steps)

        # Decoding
        if profiler:
            with torch.autograd.profiler.profile(profile_memory=True) as prof:
                with torch.autograd.profiler.record_function("Model Inference"):
                    timer = decode()
        else:
            timer = decode()

        # Profiler Print
        if profiler:
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        # Return Eval Time in s
        return timer

    def eval_time_encoder(self, dataset_eval, eval_steps=None, profiler=False):

        def forward():

            # Start Timer
            start = time.time()

            for step, batch in enumerate(eval_iterator):

                batch = [elt.to(device) for elt in batch]

                with torch.no_grad():
                    x, x_len, att = self.encoder.forward(batch[0], batch[2])

                # Evaluation Steps
                if eval_steps:
                    if step + 1 >= eval_steps:
                        break

            # Stop Timer
            return time.time() - start

        # Model Device
        device = next(self.parameters()).device

        # Evaluzation Mode
        self.eval()

        # tqdm Iterator
        eval_iterator = tqdm(dataset_eval, total=eval_steps)

        # Forward
        if profiler:
            with torch.autograd.profiler.profile(profile_memory=True) as prof:
                with torch.autograd.profiler.record_function("Model Inference"):
                    timer = forward()
        else:
            timer = forward()

        # Profiler Print
        if profiler:
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        # Return Eval Time in s
        return timer

    def eval_time_decoder(self, dataset_eval, eval_steps=None, profiler=False):

        def forward():

            # Start Timer
            start = time.time()

            for step, batch in enumerate(eval_iterator):

                batch = [elt.to(device) for elt in batch]

                hidden = None

                for i in range(batch[1].size(1)):
                    with torch.no_grad():
                        _, hidden = self.decoder.forward(batch[1][:, i:i+1], hidden)

                # Evaluation Steps
                if eval_steps:
                    if step + 1 >= eval_steps:
                        break

            # Stop Timer
            return time.time() - start

        # Model Device
        device = next(self.parameters()).device

        # Evaluzation Mode
        self.eval()

        # tqdm Iterator
        eval_iterator = tqdm(dataset_eval, total=eval_steps)

        # Forward
        if profiler:
            with torch.autograd.profiler.profile(profile_memory=True) as prof:
                with torch.autograd.profiler.record_function("Model Inference"):
                    timer = forward()
        else:
            timer = forward()

        # Profiler Print
        if profiler:
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        # Return Eval Time in s
        return timer


