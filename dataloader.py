
import torch
from torch.utils.data import Dataset, DataLoader

class BoardDataset(Dataset):
    def __init__(self, board,labels):
        self.labels = labels
        self.board = board
def __len__(self):
        return len(self.labels)
def __getitem__(self, idx):
        #todo
        label= self.labels[idx]
        board= self.board[idx]
        sample = {"board": board, "winning": label}
        return sample