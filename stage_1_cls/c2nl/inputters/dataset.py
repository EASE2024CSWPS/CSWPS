# src: https://github.com/facebookresearch/DrQA/blob/master/drqa/reader/data.py
import random

import numpy as np
from c2nl.inputters.vector import vectorize as vec
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

# ------------------------------------------------------------------------------
# PyTorch dataset class for SQuAD (and SQuAD-like) data.
# ------------------------------------------------------------------------------


class SingleDataset(Dataset):
    def __init__(self, examples, model):
        self.model = model
        self.examples = examples
        self.dicts = self.get_dict()

    def get_dict(self):
        dicts = {}
        for idx, ex in enumerate(self.examples):
            repo = ex["repo"]
            if repo not in dicts:
                dicts[repo] = []
            dicts[repo].append(idx)
        return dicts

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        repos = list(self.dicts.keys())
        exclude_repo = example["repo"]
        repos.remove(exclude_repo)
        freqs = [
            len(v) for k, v in self.dicts.items() if k != exclude_repo
        ]
        neg_repo = random.choices(repos, freqs, k=1)[0]
        neg_idx = random.choice(self.dicts[neg_repo])
        pos_list = [i for i in self.dicts[exclude_repo] if i != index]
        # pos_list.remove(index)
        try:
            pos_idx = random.choice(pos_list)
        except:
            pos_idx = index
        return vec(
            self.examples[index],
            self.examples[pos_idx],
            self.examples[neg_idx],
            self.model,
        )

    def lengths(self):
        return [(len(ex["summary"].tokens), 1) for ex in self.examples]


# ------------------------------------------------------------------------------
# PyTorch sampler returning batched of sorted lengths (by doc and question).
# ------------------------------------------------------------------------------


class SortedBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        lengths = np.array(
            [(-l[0], -l[1], np.random.random()) for l in self.lengths],
            dtype=[
                ("l1", np.int_),
                ("l2", np.int_),
                ("rand", np.float_),
            ],
        )
        indices = np.argsort(lengths, order=("l1", "l2", "rand"))
        batches = [
            indices[i : i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return len(self.lengths)
