import torch


# Dataset核心逻辑伪代码：
class Dataset(object):
    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


# DataLoader核心逻辑伪代码：
class DataLoader(object):
    def __init__(self, dataset, batch_size, collate_fn, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.sampler = torch.utils.data.RandomSampler if shuffle else \
            torch.utils.data.SequentialSampler
        self.batch_sampler = torch.utils.data.BatchSampler
        self.sample_iter = self.batch_sampler(
            self.sampler(range(len(dataset))),
            batch_size=batch_size, drop_last=drop_last)

    def __next__(self):
        indices = next(self.sample_iter)
        batch = self.collate_fn([self.dataset[i] for i in indices])
        return batch
