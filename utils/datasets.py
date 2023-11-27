import torch
import torchaudio
import glob

class AshellDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split='train'):
        self.wav_files = glob.glob(dataset_path +'/' + split + "/*/*.wav")

    def __getitem__(self, index):
        return [
            torchaudio.load(self.wav_files[index])[0],  # waveform of shape (channel, time)
            torch.load(self.wav_files[index].replace('.wav', '.pth'))
        ]

    def __len__(self):
        return len(self.wav_files)
    

def collate_fn_pad(batch):

    # Regular Mode
    if len(batch[0]) == 2:

        # Sorting sequences by lengths
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[1], reverse=True)

        # Pad data sequences
        data = [item[0].squeeze() for item in sorted_batch]
        data_lengths = torch.tensor([len(d) for d in data],dtype=torch.long) 
        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)

        # Pad labels
        target = [item[1] for item in sorted_batch]
        target_lengths = torch.tensor([t.size(0) for t in target],dtype=torch.long)
        target = torch.nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=0)

        return data, target, data_lengths, target_lengths

    else:

        raise Exception("Batch Format Error")