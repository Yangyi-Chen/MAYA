import torch.nn as nn
from transformers import BertForSequenceClassification
import abc
from torch.nn.utils.rnn import pack_padded_sequence
import torch

class VictimModel(nn.Module, metaclass=abc.ABCMeta):

    def __init__(self):
        super(VictimModel, self).__init__()

    @abc.abstractmethod
    def forward(self, arg1, arg2):
        pass


    @abc.abstractmethod
    def get_probs(self, input):
        """
        :param input: a encoded sentnece, tokenizer encoded sentence for bert & word_id str for lstm
        :return: probability distribution for each class
        """
        pass

    def get_label(self, input):
        '''

        :param input: a encoded sentnece, tokenizer encoded sentence for bert & word_id str for lstm
        :return: predict label for this sentence
        '''

        pass



class LSTM(VictimModel):
    def __init__(self, vocab_size, embed_dim=300, hidden_size=1024, layers=2, bidirectional=True, dropout=0,
                 output_class=2):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size,
                            num_layers=layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout, )
        self.linear = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_class)

    def forward(self, padded_texts, lengths):
        texts_embedding = self.embedding(padded_texts)
        packed_inputs = pack_padded_sequence(texts_embedding, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed_inputs)
        forward_hidden = hn[-1, :, :]
        backward_hidden = hn[-2, :, :]
        concat_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        output = self.linear(concat_hidden)
        return output

    def get_probs(self, input):
        '''
        :param input:
        :return: if output class = 4, then return probs, type(probs) = torch.tensor & probs.shape = (4, )
        '''

        pass
    
    def get_label(self, input):
        pass





class BERT(VictimModel):
    def __init__(self,):
        super(BERT, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased')


    def forward(self, inputs, attention_masks):
        return



    def get_probs(self, input):
        pass

    def get_label(self, input):
        pass









class Roberta(VictimModel):
    pass


