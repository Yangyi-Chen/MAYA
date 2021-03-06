import pandas as pd
import torch
import abc
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
from torch.nn.utils.rnn import pad_sequence


class NLPDataset(Dataset, metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        # iter_type means which type of dataset a dataloader will use(train, valid or test)
        super(NLPDataset, self).__init__()
        if 'iter_type' in kwargs.keys():
            self.iter_type = kwargs['iter_type']
        else:
            self.iter_type = 'train'
        self.data_types = ['train', 'valid', 'test']
        # keys of the dictionaries below are data_types
        self.iter_data_for_dataloader = {}
        self.data_for_user = {}

    def __getitem__(self, index):
        return self.iter_data_for_dataloader[self.iter_type][index]

    def __len__(self):
        return len(self.iter_data_for_dataloader[self.iter_type])

    # a subclass must call the function at the end of its __init__ function
    def initialize(self):
        self.load_dataset()
        self.load_iter_data()

    # return dataset in given type
    def get_dataset(self, type):
        return self.data_for_user[type]

    def change_iter_type(self, type):
        self.iter_type = type

    # method as a parameter in Dataloader
    @abc.abstractmethod
    def collate_fn(self):
        pass

    # method to process data for being used by dataloader
    @abc.abstractmethod
    def load_iter_data(self):
        pass

    # method to load data for users to get or being processed
    @abc.abstractmethod
    def load_dataset(self):
        pass


class SNLI_Bert(NLPDataset):
    def __init__(self, *args, **kwargs):
        super(SNLI_Bert, self).__init__(*args, **kwargs)
        self.data_tuple = namedtuple('data', ['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.initialize()

    def load_iter_data(self):
        for type in self.data_types:
            iter_data_list = []
            data = self.data_for_user[type].values
            for sample in data:
                sentence1 = self.tokenizer.tokenize(sample[0])
                sentence2 = self.tokenizer.tokenize(sample[1])
                input_tokens = ['[CLS]'] + sentence1 + ['[SEP]'] + sentence2 + ['[SEP]']
                input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
                token_type_ids = [0] * (len(sentence1) + 2) + [1] * (len(sentence2) + 1)
                attention_mask = [1] * len(input_ids)
                gold_label = sample[2]
                iter_data_list.append(self.data_tuple(torch.tensor(input_ids),
                                                      torch.tensor(token_type_ids),
                                                      torch.tensor(attention_mask),
                                                      gold_label))
                self.iter_data_for_dataloader[type] = iter_data_list

    def load_dataset(self):
        for type in self.data_types:
            dataframe = pd.read_csv('snli_' + type + '_preprocessed.tsv', sep='\t')
            self.data_for_user[type] = dataframe

    def collate_fn(self, data):
        input_ids = pad_sequence([sample.input_ids for sample in data], batch_first=True)
        token_type_ids = pad_sequence([sample.token_type_ids for sample in data], batch_first=True)
        attention_mask = pad_sequence([sample.attention_mask for sample in data], batch_first=True)
        labels = torch.tensor([sample.labels for sample in data])
        return {'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'labels': labels}



