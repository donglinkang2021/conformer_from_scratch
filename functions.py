# PyTorch
import torch

# Models
from models.model_ctc import ModelCTC

# Datasets and Preprocessing
from utils.datasets import AshellDataset, collate_fn_pad
from torch.utils.data import DataLoader

def create_model(config):
    # Create Model
    model = ModelCTC(
        encoder_params=config["encoder_params"],
        tokenizer_params=config["tokenizer_params"],
        training_params=config["training_params"],
        name=config["model_name"]
    )
    return model


def load_datasets(training_params, tokenizer_params, args):

    # Select Dataset and Split
    training_split = "train"
    evaluation_split = "dev"

    # Training Dataset
    print(training_split)
    print("Loading training dataset : {} {}".format(training_params["training_dataset"], training_split))

    dataset_train =  AshellDataset(training_params["training_dataset_path"], training_split)

    train_loader = DataLoader(
        dataset_train, 
        batch_size=training_params["batch_size"], 
        shuffle=True,  
        collate_fn=collate_fn_pad, drop_last=True, 
    )
    print("Loaded :", train_loader.dataset.__len__(), "samples", "/", train_loader.__len__(), "batches")

    # Evaluation Dataset
    print(evaluation_split)
    print("Loading evaluation dataset : {} {}".format(training_params["evaluation_dataset"], evaluation_split))

    dataset_eval = AshellDataset(training_params["evaluation_dataset_path"], evaluation_split)


    eval_loader = DataLoader(
        dataset_eval, 
        batch_size=args.batch_size_eval, 
        shuffle=True, 
        collate_fn=collate_fn_pad, 
    )
    
    print("Loaded :", eval_loader.dataset.__len__(), "samples", "/", eval_loader.__len__(), "batches")
    
    return train_loader, eval_loader