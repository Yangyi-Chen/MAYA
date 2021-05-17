import pandas as pd
import os
import torch
import tensorflow as tf
import stanza
from MG.parser import ConstituencyParser
from ADsamples_evaluation.eval_class import SentenceEncoder, GrammarChecker
from MG.rl.rl import *


class MGAttacker:
    def __init__(self,
                 attack_times=10,
                 victim_model=None,
                 substitution=None,
                 paraphrasers=None,
                 fine_tune_path=None,
                 attack_type='score',
                 save_paraphrase_label=None):

        self.victim_model = victim_model
        self.parser = ConstituencyParser()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.sim = SentenceEncoder()
        self.grammar = GrammarChecker()
        self.attack_times = attack_times
        self.fine_tune_path = fine_tune_path
        self.attack_type = attack_type
        self.save_paraphrase_label = save_paraphrase_label

        # collect supervisory singals for pretrained-agent Bert to fine-tune
        if fine_tune_path:
            if os.path.exists(fine_tune_path):
                self.samples = pd.read_csv(fine_tune_path, sep='\t')
                self.sample_num = self.samples['index'].values[-1] + 1
            else:
                self.samples = pd.DataFrame(columns=['index', 'sentence', 'sentences', 'label'])
                self.sample_num = 0

        if save_paraphrase_label:
            if os.path.exists(save_paraphrase_label):
                self.label_info = pd.read_csv(save_paraphrase_label, sep='\t')
            else:
                self.label_info = pd.DataFrame(columns=['sentence', 'phrase', 'label', 'length'])

            self.nlp = stanza.Pipeline('en', processors='tokenize,pos')

        # self.paraphrase_count = pd.DataFrame(columns=['back', 'gpt2', 'T5'])
        # self.back_count = 0
        # self.gpt2_count = 0
        # self.T5_count = 0

        self.substitution = substitution
        self.paraphrasers = paraphrasers

    # 将每个单词分别mask
    def get_masked_sentence(self, sentence):
        pos_info = None
        if self.save_paraphrase_label:
            doc = self.nlp(sentence)
            pos_info = []
            for stc in doc.sentences:
                for word in stc.words:
                    pos_info.append(word.pos)

        words = sentence.split(' ')
        masked_sentences = []

        for i in range(len(words)):
            word = words[i]
            words[i] = '[MASK]'
            tgt = ' '.join(x for x in words)
            masked_sentences.append(tgt)
            words[i] = word

        return masked_sentences, pos_info

    # 将句子统一用BertTokenizer格式化
    def formalize(self, sentences):
        formalized_sentences = []
        for ori in sentences:
            if ori is None:
                formalized_sentences.append(ori)
            else:
                tokens = self.tokenizer.tokenize(ori)

                if len(tokens) > 510:
                    tokens = tokens[0:510]

                string = self.tokenizer.convert_tokens_to_string(tokens)
                formalized_sentences.append(string)

        return formalized_sentences

    def get_best_sentences(self, sentence, paraphrases, info):
        ori_error = self.grammar.check(sentence)
        best_advs = []
        new_info = []
        # 获取同一个短语的不同paraphrase并取USE最高的
        for i in range(len(paraphrases[0])):
            advs = []

            # check the grammar error and filter out those which don't fit the restrictions
            for types in paraphrases:
                if types[i] is None:
                    continue

                adv_error = self.grammar.check(types[i])
                if adv_error <= ori_error:
                    advs.append(types[i])

            if len(advs) == 0:
                continue

            elif len(advs) == 1:
                best_advs.append(advs[0])

            else:
                # if max_index == 0:
                #     self.back_count += 1
                # elif max_index == 1:
                #     self.gpt2_count += 1
                # else:
                #     self.T5_count += 1
                best_adv = self.sim.find_best_sim(sentence, advs)[0]
                best_advs.append(best_adv)

            new_info.append(info[i])

        return best_advs, new_info

    # 对一个句子的预处理，获得所有可能的变形形式
    def sentence_process(self, sentence):
        sentence = sentence.lower()
        if self.substitution is None:
            masked_sentences, word_info = [], None
        else:
            masked_sentences, word_info = self.get_masked_sentence(sentence)

        masked_sentences = self.formalize(masked_sentences)

        root, nodes = self.parser(sentence)
        if len(nodes) == 0:
            return []

        phrases = [node[1] for node in nodes if node[3]]
        indices = [node[2] for node in nodes if node[3]]
        info = [[node[1], node[3], node[4]] for node in nodes]

        paraphrases = []
        with torch.no_grad():
            if phrases:
                for paraphraser in self.paraphrasers:
                    one_batch = paraphraser.paraphrase(phrases)
                    if one_batch is not None:
                        paraphrases.append(one_batch)

        translated_sentence_list = []
        if len(paraphrases) > 0:
            for paraphrase_list in paraphrases:
                translated_sentences = []

                for i, phrase in enumerate(paraphrase_list):
                    tree = self.parser.get_tree(phrase)
                    try:
                        root_copy = copy.deepcopy(root)
                        root_copy[indices[i]] = tree[0]
                        modified_sentence = ' '.join(word for word in root_copy.leaves()).lower()
                        translated_sentences.append(modified_sentence)

                    except Exception as e:
                        translated_sentences.append(None)

                translated_sentences = self.formalize(translated_sentences)
                translated_sentence_list.append(translated_sentences)

        best = []
        # filter out paraphrases which don't fit the grammar strict and choose the most similar one.
        if len(translated_sentence_list) > 0:
            try:
                best, info = self.get_best_sentences(sentence, translated_sentence_list, info)

            except Exception as e:
                for i in translated_sentence_list:
                    print(len(i))
                print('error in getting best paraphrases!')
                best = []

        return list(set(masked_sentences+best))

    # do one modification to given sentence
    def one_attack(self, exist_sentences, hypothesis, sentence, sentences, label):
        info_dict = dict()
        info_dict['adv'] = None
        info_dict['mask'] = None
        info_dict['done'] = False
        info_dict['substitution'] = 0
        info_dict['paraphrase'] = 0
        info_dict['query'] = 0

        with torch.no_grad():
            if hypothesis:
                inputs = [[premise, hypothesis] for premise in sentences]

            else:
                inputs = sentences

            divide = False
            while True:
                try:
                    if divide:
                        probs = torch.tensor([])
                        pred_label_list = torch.tensor([])
                        for input in inputs:
                            outputs = self.victim_model(sentences=[input])
                            probs = torch.cat((probs, outputs.probs.to('cpu')), 0)
                            pred_label_list = torch.cat((pred_label_list, outputs.pred_labels.to('cpu')), 0)

                        probs = probs[:, label]
                        pred_label_list = pred_label_list.to('cpu').numpy().tolist()

                    else:
                        outputs = self.victim_model(sentences=inputs)
                        probs = outputs.probs[:, label]
                        pred_label_list = outputs.pred_labels.to('cpu').numpy().tolist()

                    break

                except Exception as e:
                    if not divide:
                        divide = True
                    else:
                        print('attack fails: GPU out of memory!')
                        return info_dict

            paraphrase_list = []
            mask_list = []
            paraphrase_indices = []

            if self.attack_type == 'decision':
                probs = torch.tensor([self.sim.get_sim(sentence, adv) for adv in sentences])

            for i, pred_label in enumerate(pred_label_list):
                if pred_label != label:
                    if '[MASK]' in sentences[i]:
                        mask_list.append([i, sentences[i], probs[i]])

                    else:
                        paraphrase_list.append(sentences[i])
                        paraphrase_indices.append(i)

            # if attack succeeds
            if len(paraphrase_list) > 0:
                adv, index, _ = self.sim.find_best_sim(sentence, paraphrase_list)
                info_dict['adv'] = adv
                info_dict['index'] = paraphrase_indices[index]
                info_dict['done'] = True
                info_dict['paraphrase'] += 1

                return info_dict

            elif len(mask_list) > 0:
                # get sentence of least prob
                sorted(mask_list, key=(lambda x: x[2]))

                min_prob = 1
                for mask in mask_list:
                    if mask[1] in exist_sentences:
                        continue
                    else:
                        exist_sentences.append(mask[1])

                    result_dict = self.substitution.substitute(hypothesis, sentence, mask[1], label, self.attack_type)
                    info_dict['query'] += result_dict['query']
                    if not result_dict['adv'] and not result_dict['suc_advs']:
                        continue

                    if result_dict['done']:
                        adv, _, _ = self.sim.find_best_sim(sentence, result_dict['suc_advs'])
                        info_dict['mask'] = mask[1]
                        info_dict['index'] = mask[0]
                        info_dict['substitution'] += 1
                        info_dict['adv'] = adv
                        info_dict['done'] = True
                        return info_dict

                    else:
                        if self.attack_type == 'score':
                            if result_dict['prob'] < min_prob:
                                min_prob = result_dict['prob']
                                info_dict['adv'] = result_dict['adv']
                                info_dict['index'] = mask[0]
                                info_dict['mask'] = mask[1]

                        elif self.attack_type == 'decision':
                            adv, index, now_prob = self.sim.find_best_sim(sentence, result_dict['advs'], find_min=True)
                            if now_prob < min_prob:
                                min_prob = now_prob
                                info_dict['adv'] = adv
                                info_dict['index'] = index
                                info_dict['mask'] = mask[1]

                info_dict['substitution'] += 1
                return info_dict

        _, indices = torch.topk(probs, len(probs))

        # reverse the index list to get topk least probs
        indices = indices.to('cpu').numpy().tolist()
        indices.reverse()

        paraphrased_indices = []
        for i in indices:
            if '[MASK]' not in sentences[i]:
                paraphrased_indices.append(i)

        for index in indices:
            if sentences[index] in exist_sentences:
                continue
            else:
                exist_sentences.append(sentences[index])

            info_dict['index'] = index
            max_prob_sentence = sentences[index]
            if '[MASK]' in max_prob_sentence:
                result_dict = self.substitution.substitute(hypothesis,
                                                           sentence,
                                                           max_prob_sentence,
                                                           label,
                                                           self.attack_type)

                if not result_dict['adv'] and not result_dict['suc_advs']:
                    continue

                if result_dict['done']:
                    adv, _, _ = self.sim.find_best_sim(sentence, result_dict['suc_advs'])
                    info_dict['substitution'] += 1
                    info_dict['query'] += result_dict['query']
                    info_dict['adv'] = adv
                    info_dict['done'] = True
                    info_dict['mask'] = max_prob_sentence

                    return info_dict

                else:
                    if self.attack_type == 'score':
                        prob = result_dict['prob']
                        next_sentence = result_dict['adv']
                    else:
                        adv, _, prob = self.sim.find_best_sim(sentence, result_dict['advs'], find_min=True)
                        next_sentence = adv
                    times = result_dict['query']

                info_dict['query'] += times
                info_dict['adv'] = next_sentence
                info_dict['substitution'] += 1
                info_dict['mask'] = max_prob_sentence

                # 和paraphrase的句子中置信度下降最大的句子进行比较
                for i in paraphrased_indices:
                    if sentences[i] not in exist_sentences:
                        if prob > probs[i]:
                            exist_sentences.pop()
                            exist_sentences.append(sentences[i])

                            info_dict['adv'] = sentences[i]
                            info_dict['index'] = i
                            info_dict['paraphrase'] += 1
                            info_dict['substitution'] -= 1
                            info_dict['mask'] = None

                        break
                break

            else:
                info_dict['adv'] = max_prob_sentence
                info_dict['paraphrase'] += 1
                break

        return info_dict

    def attack(self, sentence, label):
        return_info = dict()

        # 统计substitution, paraphrase, query的次数
        return_info['substitution'] = 0
        return_info['paraphrase'] = 0
        return_info['query'] = 0
        return_info['adv'] = None

        hypothesis = None
        # judge whether sentence is a list(SNLI) or a string(SST2 or AGNews)
        if isinstance(sentence, list):
            premise_tokens = self.tokenizer.tokenize(sentence[0])
            hypothesis_tokens = self.tokenizer.tokenize(sentence[1])
            sentence = self.tokenizer.convert_tokens_to_string(premise_tokens)
            hypothesis = self.tokenizer.convert_tokens_to_string(hypothesis_tokens)

        else:
            tokens = self.tokenizer.tokenize(sentence)
            sentence = self.tokenizer.convert_tokens_to_string(tokens)

        # 攻击时已经出现过的句子不允许重复出现
        exist_sentences = [sentence]
        for time in range(self.attack_times):
            sentences = self.sentence_process(sentence)

            if len(sentences) == 0:
                break

            return_info['query'] += len(sentences)

            info_dict = self.one_attack(exist_sentences, hypothesis, sentence, sentences, label)

            return_info['substitution'] += info_dict['substitution']
            return_info['paraphrase'] += info_dict['paraphrase']
            return_info['query'] += info_dict['query']

            # if attack succeeds
            if self.fine_tune_path:
                for adv in sentences:
                    self.samples = self.samples.append({'index': self.sample_num,
                                                        'sentence': sentence,
                                                        'sentences': adv,
                                                        'label': info_dict['index']}, ignore_index=True)

                self.samples.to_csv(self.fine_tune_path, sep='\t', index=False)
                self.sample_num += 1

            if info_dict['done']:
                adv = info_dict['adv']
                return_info['adv'] = adv

                return return_info

            elif info_dict['adv'] is not None:
                adv = info_dict['adv']

                tokens = self.tokenizer.tokenize(adv)
                sentence = self.tokenizer.convert_tokens_to_string(tokens)

            else:
                break

        return return_info


class RLMGAttacker(MGAttacker):
    def __init__(self,
                 attack_times,
                 victim_model,
                 substitution,
                 paraphrasers,
                 agent,
                 attack_type: 'score' or 'decision'):
        super().__init__(attack_times, victim_model, substitution, paraphrasers)
        self.attack_type = attack_type
        self.agent = agent

        assert attack_type in ['score', 'decision']

    def attack(self, sentence, label):
        return_info = dict()

        # 统计substitution, paraphrase, query的次数
        return_info['substitution'] = 0
        return_info['paraphrase'] = 0
        return_info['query'] = 0
        return_info['adv'] = None

        hypothesis = None
        # judge whether sentence is a list(SNLI) or a string(SST2 or AGNews)
        if isinstance(sentence, list):
            premise_tokens = self.tokenizer.tokenize(sentence[0])
            hypothesis_tokens = self.tokenizer.tokenize(sentence[1])
            sentence = self.tokenizer.convert_tokens_to_string(premise_tokens)
            hypothesis = self.tokenizer.convert_tokens_to_string(hypothesis_tokens)

        else:
            tokens = self.tokenizer.tokenize(sentence)
            sentence = self.tokenizer.convert_tokens_to_string(tokens)

        # 攻击时已经出现过的句子不允许重复出现
        exist_sentences = [sentence]

        for time in range(self.attack_times):
            sentences = self.sentence_process(sentence)
            if len(sentences) == 0:
                break

            index = None
            if self.agent.attack_model.training:
                info_dict = self.one_attack(exist_sentences, hypothesis, sentence, sentences, label)
                if info_dict['adv'] is None:
                    break

                index = info_dict['index']

            next_sentences = self.agent(sentence, sentences, index)
            if next_sentences is None:
                return return_info

            for next_sentence in next_sentences:
                if next_sentence in exist_sentences:
                    continue
                else:
                    exist_sentences.append(next_sentence)

                if '[MASK]' in next_sentence:
                    result_dict = self.substitution.substitute(hypothesis, sentence, next_sentence, label, self.attack_type)
                    return_info['substitution'] += 1
                    return_info['query'] += result_dict['query']
                    if not result_dict['adv'] and not result_dict['suc_advs'] and not result_dict['advs']:
                        return return_info

                    if result_dict['done']:
                        adv, _, _ = self.sim.find_best_sim(sentence, result_dict['suc_advs'])
                        return_info['adv'] = adv

                        return return_info

                    elif result_dict['query'] == 0:
                        break

                    if self.attack_type == 'score':
                        next_sentence = result_dict['adv']
                    elif self.attack_type == 'decision':
                        next_sentence, _, _ = self.sim.find_best_sim(sentence, result_dict['advs'], find_min=True)

                else:
                    with torch.no_grad():
                        return_info['query'] += 1
                        return_info['paraphrase'] += 1

                        if hypothesis is None:
                            outputs = self.victim_model(sentences=next_sentence)

                        else:
                            outputs = self.victim_model(sentences=[[next_sentence, hypothesis]])

                        if outputs.pred_labels.item() != label:
                            return_info['adv'] = next_sentence

                            return return_info

                if next_sentence is None:
                    return return_info

                tokens = self.tokenizer.tokenize(next_sentence)
                sentence = self.tokenizer.convert_tokens_to_string(tokens)

                break

        # 攻击失败
        return return_info


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertForSequenceClassification.from_pretrained('bert-base-uncased').to('cuda:2')
    model = torch.nn.DataParallel(bert, device_ids=[2, 5])

    inputs = tokenizer(['you are so cool!', 'you are so cool!'], return_tensors='pt')
    result = model(**inputs)
    print(result)
    pass
