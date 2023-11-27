# Pytorch
import torch

# Functions and Utils
from functions import *

# Other
import json
import argparse
import os

def main(args):

    # Load Config
    with open(args.config_file) as json_config:
        config = json.load(json_config)

    # Device

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Device:", device)


    # Create Model
    model = create_model(config).to(device)

    vs = config["tokenizer_params"]["vocab_size"]
    print(f"vocab_size : {vs}")

    # Load Model
    if args.initial_epoch is not None:
        model.load(config["training_params"]["callback_path"] + "checkpoints_" + str(args.initial_epoch) + ".ckpt")
    else:
        args.initial_epoch = 0

    # Model Summary
    model.summary()

    # Load Dataset
    dataset_train, dataset_val = load_datasets(
        config["training_params"], 
        config["tokenizer_params"], 
        args
    )

   
    # Training
    model.fit(
        dataset_train, 
        config["training_params"]["epochs"], 
        dataset_val=dataset_val, 
        val_steps=args.val_steps, 
        verbose_val=args.verbose_val, 
        initial_epoch=int(args.initial_epoch), 
        callback_path=config["training_params"]["callback_path"], 
        steps_per_epoch=args.steps_per_epoch,
        mixed_precision=config["training_params"]["mixed_precision"],
        accumulated_steps=config["training_params"]["accumulated_steps"],
        saving_period=args.saving_period,
        val_period=args.val_period
    )



if __name__ == "__main__":

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file",          type=str,   default="ConformerCTCSmall.json",           help="Json configuration file containing model hyperparameters")
    parser.add_argument("-i", "--initial_epoch",        type=str,   default=None,                                       help="Load model from checkpoint")
    parser.add_argument("--batch_size_eval",            type=int,   default=8,                                          help="Evaluation batch size")
    parser.add_argument("--verbose_val",                action="store_true",                                            help="Evaluation verbose")
    parser.add_argument("--val_steps",                  type=int,   default=None,                                       help="Number of validation steps")
    parser.add_argument("--steps_per_epoch",            type=int,   default=None,                                       help="Number of steps per epoch")
    parser.add_argument("--cpu",                        action="store_true",                                            help="Load model on cpu")
    parser.add_argument("--saving_period",              type=int,   default=1,                                          help="Model saving every 'n' epochs")
    parser.add_argument("--val_period",                 type=int,   default=1,                                          help="Model validation every 'n' epochs")

    # Parse Args
    args = parser.parse_args()

    main(args)
