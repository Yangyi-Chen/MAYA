import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer
import abc
from torch.nn.utils.rnn import pack_padded_sequence
import torch
from collections import namedtuple

class VictimModel(nn.Module, metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        super(VictimModel, self).__init__()

    # training mode
    @abc.abstractmethod
    def train(self):
        pass

    # evaluation mode
    @abc.abstractmethod
    def eval(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def from_pretrained():
        pass

    @abc.abstractmethod
    def forward(self):
        pass

    # get model's outputs, for users to call
    @abc.abstractmethod
    def __call__(self):
        """
        :param input: a raw sentnece, need to convert the raw sentence to encoded id
        :return: probability distribution for each class, prediction of label for this sentence
        """
        pass


class LSTM(VictimModel):
    def __init__(self, vocab_size, embed_dim=300, hidden_size=1024, layers=2, bidirectional=True, dropout=0,
                 output_class=2, vocab=None):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size,
                            num_layers=layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout, )
        self.linear = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_class)
        self.vocab = vocab

    def forward(self, padded_texts, lengths):
        texts_embedding = self.embedding(padded_texts)
        packed_inputs = pack_padded_sequence(texts_embedding, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed_inputs)
        forward_hidden = hn[-1, :, :]
        backward_hidden = hn[-2, :, :]
        concat_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        output = self.linear(concat_hidden)
        return output

    def get_results(self, inputs):
        '''
        :param input:
        :return: if output class = 4, then return probs, type(probs) = torch.tensor & probs.shape = (4, )
        '''
        pass


class Victim_BertForSequenceClassification(VictimModel):
    def __init__(self, **kwargs):
        super(Victim_BertForSequenceClassification, self).__init__(**kwargs)
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', **kwargs)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.outputs = namedtuple('outputs', ['loss', 'probs', 'pred_labels'])

    def train(self):
        self.bert.train()

    def eval(self):
        self.bert.eval()

    @staticmethod
    def from_pretrained(model_file, **kwargs):
        model = Victim_BertForSequenceClassification()
        model.bert = BertForSequenceClassification.from_pretrained(model_file, **kwargs)
        return model

    def forward(self, device, input_ids, token_type_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids.to(device),
                            token_type_ids=token_type_ids.to(device),
                            attention_mask=attention_mask.to(device),
                            labels=labels.to(device))
        torch.cuda.empty_cache()
        return outputs

    def __call__(self, device='cpu', **kwargs):
        if 'sentences' in kwargs.keys():
            inputs = self.tokenizer(kwargs['sentences'], return_tensors='pt', padding=True)
        else:
            inputs = kwargs
        outputs = self.forward(**inputs, device=device)
        loss = outputs.loss
        logits = outputs.logits
        probs = torch.softmax(logits, -1)
        pred_labels = torch.argmax(probs, -1)
        return self.outputs(loss, probs, pred_labels)


class Roberta(VictimModel):
    pass




