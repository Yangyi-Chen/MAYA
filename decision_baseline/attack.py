from Attacker.ga import AttackGA
from Attacker.pso import AttackPSO
from load_utils import load_victim_model, load_dataset, load_pos_tags, load_word_candidates
from Models import *
import torch
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--Attacker', default='PSO')
parser.add_argument('--target_label', default=1, type=int)
parser.add_argument('--victimmodel_path', type=str)
parser.add_argument('--dataset_path', type=str, default='')
parser.add_argument('--postags_path', type=str, default='')
parser.add_argument('--substitue_type', default='')

params = parser.parse_args()

victim_model = load_victim_model(params.victimmodel_path)
if torch.cuda.is_available():
    victim_model.cuda()

attack_samples = load_dataset(params.dataset_path)
all_pos_tags = load_pos_tags(params.postags_path)
assert len(attack_samples) == len(all_pos_tags)

word_candidates = load_word_candidates(params.substitue_type)

attacker = eval('Attack' + params.Attacker)(victim_model, word_candidate, word_dict, max_iters=100, pop_size=60)





def encode_sentence(orig_sent):
    '''
    TODO: encode the orig_sent according to different types of victim model
    :param orig_sent:
    :return:
    '''
    pass

    return orig_sent


test_list = []
orig_list = []

adv_list = []
dist_list = []

adv_orig = []
fail_list = []

SUCCESS_THRESHOLD = 0.25

for i, orig_sample in enumerate(attack_samples):
    pos_tags = all_pos_tags[i]
    encoded_sample = encode_sentence(orig_sample)
    target_label = params.target_label
    orig_label = 0
    orig_predict = victim_model.get_probs(encoded_sample)
    if torch.argmax(orig_predict) != orig_label:
        print('skipping wrong classifed ..')
        print('--------------------------')
        continue

    x_len = torch.sum(torch.sign(encoded_sample))
    print('****** ', len(test_list) + 1, ' ********')

    test_list.append(encoded_sample)
    orig_list.append(orig_sample)

    x_adv = attacker.attack(...)

    if x_adv is None:
        print('%d failed' % (i + 1))
        fail_list.append(encoded_sample)

    else:
        num_changes = torch.sum(encoded_sample != x_adv)
        print('%d - %d changed.' % (i + 1, int(num_changes)))
        modify_ratio = num_changes / x_len

        if modify_ratio > SUCCESS_THRESHOLD:
            print('too much modification: {}, attack failed'.format(modify_ratio))
        else:
            print('successful attack')
            adv_list.append(x_adv)
            adv_orig.append(orig_sample)
            dist_list.append(modify_ratio)

    print('--------------------------')

print('Attack success rate : {:.2f}%'.format(len(adv_list) / len(test_list) * 100))
print('Median percentange of modifications: {:.02f}% '.format(
    np.median(dist_list) * 100))
print('Mean percentange of modifications: {:.02f}% '.format(
    np.mean(dist_list) * 100))


with open('AD_dpso_sem_bert.pkl', 'wb') as f:
    pickle.dump((fail_list,  adv_orig, adv_list, dist_list), f)
