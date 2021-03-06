import math
from torch.utils.data import DataLoader
import torch


class Trainer:
    def __init__(self,
                 batch_size=128,
                 optimizer=None,
                 dataset=None,
                 model=None,
                 device='cuda',
                 **kwargs):
        for key, value in locals().items():
            setattr(self, key, value)
        self.model = self.model.to(self.device)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.dataset.collate_fn)

    # one epoch of training a model
    def train(self, type='train'):
        self.model.train()
        self.dataset.change_iter_type(type)
        total = len(self.dataloader)
        count = 0
        for samples in self.dataloader:
            outputs = self.model(**samples, device=self.device)
            self.optimizer.zero_grad()
            outputs.loss.backward()
            self.optimizer.step()
            del outputs
            count += 1
            if count % 5 == 0:
                self.progress_bar(count, total)
        print('')

    # evaluate a model
    def eval(self, type='test'):
        self.model.eval()
        self.dataset.change_iter_type(type)
        count = 0
        total = 0
        progress_count = 0
        progress_total = len(self.dataloader)
        with torch.no_grad():
            for samples in self.dataloader:
                outputs = self.model(**samples, device=self.device)
                pred_labels = outputs.pred_labels.to('cpu')
                labels = samples['labels']
                total += len(pred_labels)
                count += torch.sum(pred_labels == labels).item()
                progress_count += 1
                if progress_count % 5 == 0:
                    self.progress_bar(progress_count, progress_total)
        print('')
        return count, total, count / total

    # show the progress of training
    @staticmethod
    def progress_bar(count, total):
        num = count / total * 25
        length = math.floor(num)
        progress_bar = '[' + '#' * length + ' ' * (25 - length) + ']'
        print(f'\033[0;31m\r%.2f%% {progress_bar}\033[0m' % (num * 4), end='')
