import pandas as pd
import torch
import abc
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
from torch.nn.utils.rnn import pad_sequence


class NLPDataProcessor(metaclass=abc.ABCMeta):
    """
    This class provides different ways of data processing for different model such as BERT, BILSTM, etc.
    you need to implement the function below to make sure it could be smoothly called by class
    Dataloader.

    Class NLPDataset would process its dataset based on the function below for Dataloader.

    """
    @abc.abstractmethod
    def collate_fn(self):
        raise Exception("Abstract method ""collate_fn"" method not be implemented!")


class BertProcessor(NLPDataProcessor):
    def collate_fn(self, data):
        input_ids = pad_sequence([sample.input_ids for sample in data], batch_first=True)
        token_type_ids = pad_sequence([sample.token_type_ids for sample in data], batch_first=True)
        attention_mask = pad_sequence([sample.attention_mask for sample in data], batch_first=True)
        labels = torch.tensor([sample.labels for sample in data])
        return {'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'labels': labels}


class NLPDataset(Dataset, metaclass=abc.ABCMeta):

    def __init__(self, *args, **kwargs):
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

    def initialize(self):
        self.load_dataset()

    # return dataset in given type
    def get_dataset(self, type):
        return self.data_for_user[type]

    def change_iter_type(self, type):
        self.iter_type = type

    # method to load data for users to get or being processed
    @abc.abstractmethod
    def load_dataset(self):
        raise Exception("Abstract method ""load_dataset"" method not be implemented!")

    # method as a parameter in Dataloader
    @abc.abstractmethod
    def collate_fn(self):
        raise Exception("Abstract method ""collate_fn"" method not be implemented!")

    # method to process data for being used by dataloader
    @abc.abstractmethod
    def load_iter_data(self):
        raise Exception("Abstract method ""load_iter_data"" method not be implemented!")


# Dataset class, implements the function load_dataset
# doesn't implement the function load_iter_data and collate_fn, just for users to get dataset
class SNLI(NLPDataset):
    def __init__(self, *args, **kwargs):
        super(SNLI, self).__init__(*args, **kwargs)
        super(SNLI, self).initialize()

    def initialize(self):
        self.load_iter_data()

    def load_dataset(self):
        for type in self.data_types:
            dataframe = pd.read_csv('snli_' + type + '_preprocessed.tsv', sep='\t')
            self.data_for_user[type] = dataframe

    def collate_fn(self):
        super().collate_fn()

    # method to process data for being used by dataloader
    def load_iter_data(self):
        super().load_iter_data()


class SST2(NLPDataset):
    def __init__(self, *args, **kwargs):
        super(SST2, self).__init__(*args, **kwargs)
        super(SST2, self).initialize()

    def initialize(self):
        self.load_iter_data()

    def collate_fn(self):
        super().collate_fn()

    # method to process data for being used by dataloader
    def load_iter_data(self):
        super().load_iter_data()

    def load_dataset(self):
        for type in self.data_types:
            dataframe = pd.read_csv('sst2_' + type + '.tsv', sep='\t')
            self.data_for_user[type] = dataframe


# implement the function load_iter_data and collate_fn, could be called by Dataloader.
class SNLI_Bert(BertProcessor, SNLI):
    def __init__(self, *args, **kwargs):
        super(SNLI_Bert, self).__init__(*args, **kwargs)
        super(SNLI_Bert, self).initialize()

    def load_iter_data(self):
        data_tuple = namedtuple('data', ['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        for type in self.data_types:
            iter_data_list = []
            data = self.data_for_user[type].values
            for sample in data:
                sentence1 = tokenizer.tokenize(sample[0])
                sentence2 = tokenizer.tokenize(sample[1])
                input_tokens = ['[CLS]'] + sentence1 + ['[SEP]'] + sentence2 + ['[SEP]']
                input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                token_type_ids = [0] * (len(sentence1) + 2) + [1] * (len(sentence2) + 1)
                attention_mask = [1] * len(input_ids)
                gold_label = sample[2]
                iter_data_list.append(data_tuple(torch.tensor(input_ids),
                                                 torch.tensor(token_type_ids),
                                                 torch.tensor(attention_mask),
                                                 gold_label))
                self.iter_data_for_dataloader[type] = iter_data_list

    def collate_fn(self, data):
        return super().collate_fn(data)


class SST2_Bert(BertProcessor, SST2):
    def __init__(self, *args, **kwargs):
        super(SST2_Bert, self).__init__(*args, **kwargs)
        super(SST2_Bert, self).initialize()

    def load_iter_data(self):
        data_tuple = namedtuple('data', ['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        for type in self.data_types:
            data = self.data_for_user[type].values
            sentences = [sample[0] for sample in data]
            labels = [sample[1] for sample in data]
            iter_data_list = []
            tokenized_sentences = tokenizer(sentences)
            for i, label in enumerate(labels):
                iter_data_list.append(data_tuple(torch.tensor(tokenized_sentences['input_ids'][i]),
                                                 torch.tensor(tokenized_sentences['token_type_ids'][i]),
                                                 torch.tensor(tokenized_sentences['attention_mask'][i]),
                                                 label))
            self.iter_data_for_dataloader[type] = iter_data_list

    def collate_fn(self, data):
        return super().collate_fn(data)


# Provide a method to load different darasets provided.
def load(dataset, **kwargs):
    if dataset == 'SNLI_Bert':
        return SNLI_Bert(kwargs)


if __name__ == '__main__':
    sst2 = SST2_Bert(iter_type='test')
    dataset = sst2.get_dataset('test')
    dataloader = DataLoader(sst2, batch_size=2, collate_fn=sst2.collate_fn)
    for samples in dataloader:
        print(samples)
        break

