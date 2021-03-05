import pandas as pd
import copy
import torch
import math
import os
from transformers import BertTokenizer, BertForSequenceClassification, BertForMaskedLM
from BaiduTransAPI_forPython3 import Translator
from nltk.tree import Tree
from supar import Parser


class ConstituencyParser:
    def __init__(self):
        self.parser = Parser.load('crf-con-bert-en')

    def get_tree(self, input):
        word_list = input.replace('(', ',').replace(')', ',').split(' ')
        for word in word_list:
            if word == '':
                word_list.remove(word)
        prediction = self.parser.predict(word_list, verbose=False)
        return prediction.trees[0]

    def __call__(self, input):
        # 返回句子的每个节点及索引
        word_list = input.replace('(', ',').replace(')', ',').split(' ')
        for word in word_list:
            if word == '':
                word_list.remove(word)
        trees = self.parser.predict(word_list, verbose=False)
        root = trees.trees[0]
        node_list = pd.DataFrame(columns=['sub_tree', 'phrase', 'index'])
        for index in root.treepositions():
            sub_tree = root[index]
            if isinstance(sub_tree, Tree):
                if len(sub_tree.leaves()) > 1:
                    phrase = ' '.join(word for word in sub_tree.leaves())
                    node_list = node_list.append({'sub_tree': sub_tree,
                                                  'phrase': phrase,
                                                  'index': index}, ignore_index=True)
        node_list = node_list.drop_duplicates('phrase', keep='last')
        return root, node_list.values


# 将每个单词分别mask
def get_masked_sentence(sentence):
    words = sentence.split(' ')
    masked_sentences = []
    for i in range(len(words)):
        words_copy = copy.deepcopy(words)
        words_copy[i] = '[MASK]'
        masked_sentences.append(' '.join(word for word in words_copy))
    return masked_sentences


# 对一个句子的预处理，获得所有可能的变形形式
def sentence_process(sentence):
    masked_sentences = get_masked_sentence(sentence)
    root, nodes = parser(sentence)
    phrases = [node[1] for node in nodes]
    translations = translator.translate('en', 'zh', phrases)
    back_translations = translator.translate('zh', 'en', translations)
    translated_sentences = [sentence]
    # 用译回的子树替换原先的子树
    for i, phrase in enumerate(back_translations):
        tree = parser.get_tree(phrase)
        root_copy = copy.deepcopy(root)
        root_copy[nodes[i][2]] = tree[0]
        modified_sentence = ' '.join(word for word in root_copy.leaves())
        if modified_sentence not in translated_sentences:
            translated_sentences.append(modified_sentence)
    return masked_sentences + translated_sentences


# 用Bert模型预测MASK位置的单词并选择攻击强度最大的句子返回
def get_predict_sentence(sentence, label):
    inputs = tokenizer(sentence, return_tensors='pt')
    index = None
    tokenized_sentence = inputs['input_ids']
    for i in range(tokenized_sentence.size()[1]):
        if tokenized_sentence[0][i] == 103:
            index = i
    with torch.no_grad():
        outputs = predictor(input_ids=inputs['input_ids'].to(device),
                            token_type_ids=inputs['token_type_ids'].to(device),
                            attention_mask=inputs['attention_mask'].to(device))
    logits = torch.softmax(outputs.logits[0][index], -1)
    probs, indices = torch.topk(logits, 10)
    modified_sentences = torch.empty(10, tokenized_sentence.size()[1], dtype=torch.long)
    for i, location in enumerate(indices):
        tokenized_sentence[0][index] = location
        modified_sentences[i] = tokenized_sentence[0]
    token_type_ids = torch.zeros_like(modified_sentences)
    attention_mask = torch.ones_like(modified_sentences)
    outputs = victim_model(input_ids=modified_sentences.to(device),
                           token_type_ids=token_type_ids.to(device),
                           attention_mask=attention_mask.to(device))
    prob = torch.softmax(outputs.logits, -1)
    index = torch.argmin(prob, 0)[label]
    tokens = tokenizer.convert_ids_to_tokens(modified_sentences[index][1:-1])
    return prob[index][0 if label == 1 else 1], tokenizer.convert_tokens_to_string(tokens)


def attack(inputs, labels):
    if os.path.exists('attack_test_samples.tsv'):
        attack_samples = pd.read_csv('attack_test_samples.tsv', sep='\t')
    else:
        attack_samples = pd.DataFrame(columns=['attack_sample', 'substitution', 'paraphrase', 'query'])
    total = len(inputs)
    count = len(attack_samples)
    for i in range(count, total):
        sentence = inputs[i]
        # 统计substitution, paraphrase, query的次数
        substitute_count = 0
        paraphrase_count = 0
        query_times = 0
        exist_sentences = [sentence]
        while True:
            sentences = sentence_process(sentence)
            query_times += len(sentences)
            tokenized_inputs = tokenizer(sentences, padding=True, return_tensors='pt')
            with torch.no_grad():
                outputs = victim_model(input_ids=tokenized_inputs['input_ids'].to(device),
                                       token_type_ids=tokenized_inputs['token_type_ids'].to(device),
                                       attention_mask=tokenized_inputs['attention_mask'].to(device))
            prob = torch.softmax(outputs.logits, -1)
            wrong_prob = prob[:, 0 if labels[i] == 1 else 1]
            _, indices = torch.topk(wrong_prob, len(wrong_prob))
            for index in indices:
                prob = wrong_prob[index]
                max_prob_sentence = sentences[index]
                if '[MASK]' in max_prob_sentence:
                    query_times += 10
                    prob, next_sentence = get_predict_sentence(max_prob_sentence, labels[i])
                    if next_sentence in exist_sentences:
                        # 如果更改后的句子之前已经出现过，则按置信度下降排序选择新的句子去更改
                        continue
                    else:
                        sentence = next_sentence
                        exist_sentences.append(sentence)
                        substitute_count += 1
                else:
                    if max_prob_sentence in exist_sentences:
                        continue
                    else:
                        sentence = max_prob_sentence
                        exist_sentences.append(sentence)
                        paraphrase_count += 1
                break
            if prob > 0.5:
                attack_samples = attack_samples.append({'attack_sample': sentence,
                                                        'substitution': substitute_count,
                                                        'paraphrase': paraphrase_count,
                                                        'query': query_times},
                                                       ignore_index=True)
                break
        # 展示训练进度
        count += 1
        num = count / total * 25
        progress_bar = '[' + '#' * math.floor(num) + ' ' * (25 - math.floor(num)) + ']'
        print(f'\033[0;31m\r攻击进度：{count}/{total} %.2f%% {progress_bar}\033[0m' % (num*4), end='')
        if count % 20 == 0:
            attack_samples.to_csv('attack_test_samples.tsv', sep='\t', index=False)


def ave_modify_times():
    attack_samples = pd.read_csv('attack_test_samples.tsv', sep='\t')
    substitution = attack_samples['substitution'].values
    paraphrase = attack_samples['paraphrase'].values
    query = attack_samples['query'].values
    total = len(substitution)
    ave_substitution = torch.sum(torch.tensor(substitution)) / total
    ave_paraphrase = torch.sum(torch.tensor(paraphrase)) / total
    ave_query = torch.sum(torch.tensor(query)) / total
    print('总攻击样本数：{}，平均substitution次数：{:.2f}，'
          '平均paraphrase次数：{:.2f}，'
          '平均query次数：{:.2f}'.format(total, ave_substitution, ave_paraphrase, ave_query))


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    victim_model = BertForSequenceClassification.from_pretrained('BertForSST_2').to(device)
    predictor = BertForMaskedLM.from_pretrained('BertForMaskedLM').to(device)
    parser = ConstituencyParser()
    translator = Translator()
    train_set = pd.read_csv('correct_test.tsv', sep='\t')
    samples = train_set['sentence'].values
    labels = train_set['label'].values
    attack(samples, labels)

