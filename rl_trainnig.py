import os
import pandas as pd
import torch.optim as optim
import torch
from MG.multi_granularity import RLMGAttacker, MGAttacker
from models.models import *
from transformers import BertForSequenceClassification
from rl import Agent
from MG.paraphrase_methods import *
from MG.substitute_methods import SubstituteWithBert
from tqdm import tqdm


def train():
    os.environ['CUDA_VISIBLE_DEVICES'] = "6,7"
    data_type = 'sst2'
    model_type = 'Roberta'
    output_class = 2
    print(data_type, model_type)
    vocab_path = f'../../datasets/{data_type}/{data_type}_vocab.tsv'

    if model_type == 'BiLSTM':
        victim_model_path = f'../../models/pretrained_models/bilstm_for_{data_type}_for_rl.pth'
    else:
        victim_model_path = f'../../models/pretrained_models/{model_type.lower()}_for_{data_type}_for_rl'

    dest_model_path = f'../../models/pretrained_models/mayapi_t5_{model_type.lower()}_for_{data_type}'
    save_model_path = f'../../models/pretrained_models/mayapi_t5_{model_type.lower()}_for_{data_type}'

    dataset = pd.read_csv(f'../../datasets/{data_type}/{model_type}/origin/'
                          f'{data_type}_{model_type}_correct_train_rl.tsv', sep='\t')

    if model_type == 'BiLSTM':
        if data_type == 'mnli':
            victim_model = VictimLSTMForTwoSeq(vocab_path=vocab_path, output_class=output_class).to('cuda:0')
        else:
            victim_model = VictimLSTMForOneSeq(vocab_path=vocab_path, output_class=output_class).to('cuda:0')
        victim_model.load_state_dict(torch.load(victim_model_path))
        victim_model.eval()

    elif model_type == 'Roberta':
        victim_model = VictimRobertaForSequenceClassification(victim_model_path).to('cuda:0')
        victim_model.eval()

    else:
        victim_model = VictimBertForSequenceClassification(victim_model_path).to('cuda:0')
        victim_model.eval()

    if os.path.exists(dest_model_path):
        attack_model = BertForSequenceClassification.from_pretrained(dest_model_path, num_labels=1)
    else:
        attack_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

    attack_model = torch.nn.DataParallel(attack_model, device_ids=[1, 0])
    attack_model.to('cuda:1')

    optimizer = optim.Adam(attack_model.parameters(), lr=2e-5)
    optimizer.zero_grad()

    agent = Agent(attack_model, optimizer, 16)

    substitution = SubstituteWithBert(victim_model, 'cuda:0')
    # back_tran = BackTranslation()
    # gpt2 = GPT2Paraphraser(gpt2_path, 'cuda:3')
    t5 = T5('cuda:0')

    attacker = RLMGAttacker(10, victim_model, substitution,  [t5], agent, 'score')

    epochs = 100
    agent.attack_model.train()

    loss_path = f'../../ADsamples_evaluation/{data_type}_{model_type.lower()}_t5_rl_loss_test.tsv'
    if os.path.exists(loss_path):
        loss_data = pd.read_csv(loss_path, sep='\t')
        count = loss_data.shape[0]
    else:
        loss_data = pd.DataFrame(columns=['times', 'loss'])
        count = 0

    for epoch in range(epochs):
        data = dataset.sample(frac=1).values
        print('epoch{}:'.format(epoch))
        for i, sample in tqdm(enumerate(data), desc=f'{data_type} {model_type}'):
            if data_type == 'mnli':
                sentence = [sample[0], sample[1]]
            else:
                sentence = sample[0]
            label = sample[-1]

            attacker.attack(sentence, label)

            # train
            if attacker.agent.is_full():
                count += 1
                loss = attacker.agent.train()
                print('in {} times training, loss = {}'.format(count, loss))
                loss_data = loss_data.append({'times': count,
                                              'loss': loss}, ignore_index=True)

                loss_data.to_csv(loss_path, sep='\t', index=False)
                agent.attack_model.module.save_pretrained(save_model_path)


if __name__ == '__main__':
    train()
    # attack_model = BertForSequenceClassification.from_pretrained('model-base-uncased', num_labels=1)
    # attack_model.save_pretrained('../../models/pretrained_models/mgrl_for_be1rt')
    pass