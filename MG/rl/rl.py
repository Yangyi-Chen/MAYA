import pandas as pd
import copy
import torch
import random
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertForSequenceClassification

class Agent:
    def __init__(self, attack_model, optimizer=None, batch_size=64):
        self.attack_model = attack_model
        self.optimizer = optimizer
        self.batch_size = batch_size

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # check if number of losses has increased to batch_size
        self.buffer_count = 0
        # weight used for computing correct loss
        self.total_count = 0
        # to compute loss
        self.loss_func = torch.nn.CrossEntropyLoss()
        # output final average loss
        self.loss = 0

    def is_full(self):
        if self.buffer_count >= self.batch_size:
            return True
        else:
            return False

    def train(self):
        if self.buffer_count == 0:
            return

        # divide grad by total_count for every parameter
        for group in self.optimizer.param_groups:
            for p in group['params']:
                p.grad /= self.total_count

        self.optimizer.step()
        self.optimizer.zero_grad()

        loss = self.loss / self.total_count
        self.loss = 0
        self.buffer_count = 0
        self.total_count = 0

        return loss

    def __call__(self, ori_sentence, candidates, label):
        """
        :param ori_sentence: sentence to be modified
        :param candidates: choose the best one to take the place of original sentence
        :return: chosen sentence

        """
        if self.attack_model.training:
            torch.set_grad_enabled(True)
        else:
            torch.set_grad_enabled(False)

        input_ids = []
        token_type_ids = []
        attention_mask = []

        for candidate in candidates:
            inputs = self.tokenizer(ori_sentence, candidate, return_token_type_ids=True)

            input_ids.append(torch.tensor(inputs['input_ids']))
            token_type_ids.append(torch.tensor(inputs['token_type_ids']))
            attention_mask.append(torch.tensor(inputs['attention_mask']))

        device = next(self.attack_model.parameters()).device
        input_ids = pad_sequence(input_ids, batch_first=True)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)

        # restrict max input length to 512
        if input_ids.size()[1] > 512:
            input_ids = input_ids[:, 0:512]
            token_type_ids = token_type_ids[:, 0:512]
            attention_mask = attention_mask[:, 0:512]

        divide = False
        while True:
            try:
                if divide:
                    logits = torch.tensor([]).to(device)
                    step = len(self.attack_model.device_ids) if isinstance(self.attack_model, torch.nn.DataParallel) else 1
                    batch_size = input_ids.size()[0]
                    for i in range(0, batch_size, step):
                        end = i + step if i + step <= batch_size else batch_size
                        outputs = self.attack_model(input_ids=input_ids[i:end].to(device),
                                                    token_type_ids=token_type_ids[i:end].to(device),
                                                    attention_mask=attention_mask[i:end].to(device))
                        logits = torch.cat((logits, outputs.logits), 0)

                else:
                    outputs = self.attack_model(input_ids=input_ids.to(device),
                                                token_type_ids=token_type_ids.to(device),
                                                attention_mask=attention_mask.to(device))
                    logits = outputs.logits
                break

            except Exception as e:
                if not divide:
                    divide = True
                else:
                    return None

        _, rank = torch.sort(logits.to('cpu'), 0, descending=True)

        # compute grad
        if self.attack_model.training and self.buffer_count < self.batch_size:
            logits = logits.resize(1, len(candidates))
            loss = self.loss_func(logits, torch.tensor([label]).to(device))
            loss *= len(candidates)
            try:
                loss.backward()
                self.loss += loss.item()
                self.buffer_count += 1
                self.total_count += len(candidates)

            except Exception:
                print("GPU out of memory when doing backward propagation, do gradient descent in advance")
                self.train()

        return [candidates[i] for i in rank]


if __name__ == '__main__':
    from models.models import VictimBertForSequenceClassification

    res = torch.tensor([[0.5], [0.4], [0.6]])
    sentences = ['gagag', 'whwhw', 'wahawae']
    values, indices = torch.sort(res, 0, descending=True)
    print(values, indices)
    sentences = [sentences[i] for i in indices]
    print(sentences)
    pass
