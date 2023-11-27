from collections import Counter

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self, reserved_tokens=None):
        if reserved_tokens is None:
            reserved_tokens = []
        self.idx_to_token = reserved_tokens + ['<unk>']
        self.token_to_idx = {
            token: idx for idx, token in enumerate(self.idx_to_token)
        }

    def add_token(self, token):
        if token not in self.token_to_idx:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1    

    def __len__(self):
        return len(self.idx_to_token)
    
    def __call__(self, words):
        """
        see self.to_indices
        """
        return self.to_indices(words)
            
    
    def to_tokens(self, indices):
        """
        Convert indices to tokens.
        @param indices (list[int]) or (int) : a list of indices or a single index
        @returns tokens (list[str]) : a list of tokens corresponding to indices
        """
        return [self.idx_to_token[index] for index in indices]
    
    def to_indices(self, tokens):
        """
        Convert tokens to indices. 
        @param tokens (list[str]) or (str) : a list of tokens, a single token or a sentence
        @returns indices (list[int]) : a list of indices corresponding to tokens
        """
        return [self.token_to_idx[token] if token in self.token_to_idx else self.token_to_idx['<unk>'] for token in tokens]
    
    
    def save(self, file_path):
        # 清空原来的vocab文件
        with open(file_path, 'w') as f:
            f.seek(0)
            f.truncate() 
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(str(self.token_to_idx))

    @property
    def unk(self):  # Index for the unknown token
        return 1

    @property
    def token_freqs(self):  # List of (token, freq) pairs
        return self._token_freqs

    @staticmethod
    def load(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            vocab_dict = eval(f.readlines()[0]) # 只有一行数据，所以读取第一行数据
        vocab = Vocabulary()
        vocab.token_to_idx = vocab_dict
        vocab.idx_to_token = list(vocab_dict.keys())
        return vocab
    

def build_vocab(data_path, min_freq=0, reserved_tokens=['<pad>']):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    vocab = Vocabulary(
        reserved_tokens = reserved_tokens
    )
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        content = line.split('\t')[-1].replace('\n', '')
        for word in content:
            counter.update(word)
    
    _token_freqs = sorted(
        counter.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for token, freq in _token_freqs:
        if freq < min_freq:
            break
        vocab.add_token(token)
    
    return vocab


if __name__ == '__main__':
    vocab = build_vocab(data_path=f'./data/aishell/transcript/aishell_transcript_v0.8.txt')
    vocab.save('./data/aishell/vocab.txt')
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to 'data/process/vocab.txt'")