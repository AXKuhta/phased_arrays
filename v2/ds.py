
from torch.utils.data import Dataset, DataLoader
import torch

class DirectionSamples(Dataset):
	prng = torch.Generator()
	prng.manual_seed(42)

	def __init__(self):
		pass

	def __len__(self):
		return 5_000_000

	def __getitem__(self, idx):
		tp = torch.rand(2, generator=self.prng, dtype=torch.float32)
		tp[0] = torch.arccos(tp[0])
		tp[1] = tp[1]*torch.pi*2
		return tp

dataset = DirectionSamples()
train_dataloader = DataLoader(dataset, batch_size=64)
