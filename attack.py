import pandas as pd
import os
from MG.rl import Agent
from models.models import VictimBertForSequenceClassification
from MG.multi_granularity import MGAttacker
from MG.paraphrase_methods import T5, GPT2Paraphraser, BackTranslation
from MG.substitute_methods import SubstituteWithBert
from tqdm import tqdm


def attack():
    if os.path.exists(dest_path):
        attack_samples = pd.read_csv(dest_path, sep='\t')
        count = attack_samples['index'].values[-1] + 1
    else:
        attack_samples = pd.DataFrame(columns=['index', 'ori', 'adv', 'substitution', 'paraphrase', 'query'])
        count = 0

    total = len(samples)
    print(total)
    print(count)
    # show the information about attacking
    progress_bar = tqdm(range(count, total))
    suc = len(attack_samples.values)
    fail = count - suc
    attack_num = count
    suc_rate = suc / attack_num * 100 if attack_num != 0 else 0

    for i in progress_bar:
        progress_bar.set_description(
            '\033[0;31mparaphase:{} suc:{}  fail:{} total:{}  suc_rate:{:.2f}%\033[0m'.format(
                paraphrases, suc, fail, attack_num, suc_rate))

        info = attacker.attack(samples[i], labels[i])

        if info['adv'] is None:
            fail += 1
            attack_num += 1
            suc_rate = suc / attack_num * 100

        else:
            suc += 1
            attack_num += 1
            suc_rate = suc / attack_num * 100
            ori = samples[i][0] if isinstance(samples[i], list) else samples[i]
            attack_samples = attack_samples.append({'index': i,
                                                    'ori': ori,
                                                    'adv': info['adv'],
                                                    'substitution': info['substitution'],
                                                    'paraphrase': info['paraphrase'],
                                                    'query': info['query']}, ignore_index=True)

        if attack_samples.shape[0] > 0:
            attack_samples.to_csv(dest_path, sep='\t', index=False)
            pass


if __name__ == '__main__':
    # fill your own dataset path
    print("load dataset")
    dataset = pd.read_csv('datasets/sst2.tsv', sep='\t')

    samples = dataset['sentence'].values
    # samples = dataset[['sentence1', 'sentence2']].values
    labels = dataset['label'].values

    # fill your own victim model path
    print("load victim model")
    victim_model = VictimBertForSequenceClassification('models/pretrained_models/bert_for_sst2')

    # choose your paraphrase models
    paraphrases = ['t5']
    paraphrase_list = []

    if 'bt' in paraphrases:
        # use back translation
        print("initialize back tran")
        paraphrase_list.append(BackTranslation())

    if 't5' in paraphrases:
        # use T5
        print("initialize T5")
        paraphrase_list.append(T5('cuda:0'))

    if 'gpt2' in paraphrases:
        # use gpt2 paraphrase model
        # fill your own gpt2 model path
        print("initialize gpt2")
        gpt2_path = 'paraphrase_models/style_transfer_paraphrase/paraphraser_gpt2_large'
        paraphrase_list.append(GPT2Paraphraser(gpt2_path, 'cuda:1'))

    # choose your paraphrase models
    print("initialize substitution model")
    substitution = SubstituteWithBert(victim_model, 'cuda:0')

    # output result
    dest_path = './results.tsv'

    attack_times = 10

    print("initialize attacker")
    attacker = MGAttacker(attack_times, victim_model, substitution, paraphrase_list)

    # start attack
    print("start attack")
    attack()
