from transformers import BertTokenizer, BertForTokenClassification, BertForMaskedLM, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import pandas as pd
import random
import copy
import math
from nltk.corpus import wordnet as wn


class DQN:
    def __init__(self, model, buffer_size, batch_size, interval):
        self.model = model
        self.fixed_model = copy.deepcopy(model)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.interval = interval
        self.count = 0
        self.buffer = pd.DataFrame(columns=['state', 'action', 'reward', 'next_state'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)

    def store_into_buffer(self, states, actions, rewards, next_states, done):
        # 如果缓冲区满，则随机替换掉原先缓冲区中的若干数据
        if self.buffer.shape[0] + len(states) > self.buffer_size:
            remove_num = self.buffer.shape[0] + len(states) - self.buffer_size
            # 随机选择被替换的数据
            indices = random.sample(range(0, self.buffer.shape[0]), remove_num)
            self.buffer = self.buffer.drop(indices)
        for i in range(len(states)):
            self.buffer = self.buffer.append({'state': states[i],
                                              'action': actions[i],
                                              'reward': rewards[i],
                                              'next_state': next_states[i],
                                              'done': done[i]}, ignore_index=True)

    def load_from_buffer(self):
        # 如果缓冲区里的数据小于batch_size，直接全部取出
        if len(self.buffer) <= self.batch_size:
            samples = self.buffer
        else:
            # 从缓冲区里随机抽样出一个batch_size的数据用来训练
            samples = self.buffer.sample(self.batch_size)
        return [samples[column].values for column in samples.columns]

    def get_Qscores(self, states, model, lengths=None, actions=None, done=None):
        token_type_ids = [torch.zeros_like(state) for state in states]
        attention_masks = [torch.ones_like(state) for state in states]
        # 用PAD进行填充
        input_ids = pad_sequence(states, batch_first=True)
        attention_mask = pad_sequence(attention_masks, batch_first=True)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True)
        outputs = model(input_ids=input_ids.to(device),
                        token_type_ids=token_type_ids.to(device),
                        attention_mask=attention_mask.to(device))
        logits = outputs.logits.squeeze(-1)
        if actions is not None:
            Q_scores = [logits[i][action] for i, action in enumerate(actions)]
            return Q_scores
        else:
            # 如果句子本身对模型已经攻击成功，则下一次改变后的reward只能为0
            Q_scores = [torch.max(logits[i][1:length-1])
                        if done[i].item() is False else 0 for i, length in enumerate(lengths)]
            return Q_scores

    def train(self):
        self.model.train()
        states, actions, rewards, next_states, done = self.load_from_buffer()
        lengths = [len(state) for state in states]
        Q_scores = self.get_Qscores(states,
                                    model=self.model,
                                    actions=actions)
        Q_next_scores = self.get_Qscores(next_states,
                                         model=self.fixed_model,
                                         lengths=lengths,
                                         done=done)
        targets = torch.tensor(Q_next_scores) + torch.tensor(rewards)
        # 利用MSE计算loss
        loss = 0
        for i, Q_score in enumerate(Q_scores):
            loss += torch.square(Q_score - targets[i])
        loss /= len(Q_scores)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # 每达到一定间隔就更新fixed_model
        self.count += 1
        if self.count == self.interval:
            self.fixed_model = copy.deepcopy(self.model)


# 情感分析Dataset类
class sst2_Dataset(Dataset):
    def __init__(self, dataset, tokenizer):
        # dataset是用pandas读取的DataFrame类，tokenizer是BertTokenizer
        self.len = len(dataset)
        # 将dataset中的句子和标签分开
        sentences = [sample[0] for sample in dataset.values]
        self.labels = [sample[1] for sample in dataset.values]
        # 将所有句子都进行tokenize并保存
        tokenized_sentences = tokenizer(sentences)
        self.input_ids = [torch.tensor(sample) for sample in tokenized_sentences['input_ids']]
        self.token_type_ids = [torch.tensor(sample) for sample in tokenized_sentences['token_type_ids']]
        self.attention_mask = [torch.tensor(sample) for sample in tokenized_sentences['attention_mask']]
        self.lengths = [len(sample) for sample in tokenized_sentences['input_ids']]

    def __getitem__(self, index):
        return self.input_ids[index], \
               self.token_type_ids[index], \
               self.attention_mask[index], \
               self.lengths[index], \
               self.labels[index]

    def __len__(self):
        return self.len


# DataLoader类的函数参数
def collate_fn(data):
    input_ids = [single_data[0] for single_data in data]
    token_type_ids = [single_data[1] for single_data in data]
    attention_masks = [single_data[2] for single_data in data]
    lengths = torch.tensor([single_data[3] for single_data in data])
    labels = torch.tensor([single_data[4] for single_data in data])
    # 利用zero_padding填充不等长的句子
    input_ids = pad_sequence(input_ids, batch_first=True)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True)
    attention_masks = pad_sequence(attention_masks, batch_first=True)
    return input_ids, token_type_ids, attention_masks, lengths, labels


# 与环境交互的模型，即判断如何修改句子
class Agent:
    def __init__(self, attack_model, epsilon):
        self.attack_model = attack_model
        self.epsilon = self.epsilon_value = epsilon

    def train(self):
        self.epsilon = self.epsilon_value

    def eval(self):
        self.epsilon = 1

    # 对于给定的一批states，给出相应的actions
    def get_actions(self, states):
        self.attack_model.eval()
        with torch.no_grad():
            outputs = self.attack_model(input_ids=states[0].to(device),
                                        token_type_ids=states[1].to(device),
                                        attention_mask=states[2].to(device))
        logits = outputs.logits.squeeze(-1)
        lengths = states[3]
        actions = []
        for i, sentence in enumerate(logits):
            # 有一定几率随机选择一个位置修改
            if random.random() > self.epsilon:
                actions.append(random.randint(1, lengths[i]-2))
            else:
                actions.append(torch.argmax(sentence[1:lengths[i]-1]).item() + 1)
        return actions


# 与agent交互的环境，即受害模型
class Env:
    def __init__(self, model, modifier):
        self.model = model
        self.modifier = modifier

    def get_prob(self, states):
        with torch.no_grad():
            outputs = self.model(input_ids=states[0].to(device),
                                 token_type_ids=states[1].to(device),
                                 attention_mask=states[2].to(device))
        prob = torch.softmax(outputs.logits, -1)
        return prob.to('cpu')

    def step(self, states, actions):
        labels = states[-1]
        # 获取原来的预测概率
        original_prob = self.get_prob(states)
        self.modifier(states, actions)
        observation = copy.deepcopy(states[0])
        # 修改句子状态后的预测概率
        next_prob = self.get_prob(states)
        # for ob in observation:
        #     tokens = tokenizer.convert_ids_to_tokens(ob)
        #     sentence = tokenizer.convert_tokens_to_string(tokens)
        #     print(sentence)
        # 判断标签是否改变
        _, prediction = torch.max(next_prob, 1)
        done = prediction != labels
        not_done = prediction == labels
        rewards = []
        for i, label in enumerate(labels):
            # 以置信度的下降作为奖赏
            reward = (original_prob[i][label]-next_prob[i][label])
            # 如果模型的判断改变了，则增加高额奖赏
            if done[i]:
                reward += 10
            rewards.append(reward.item())
        # 将已经攻击完成的样本剔除
        for i, element in enumerate(states):
            states[i] = element[not_done]
        return observation, rewards, done


# 对句子进行修改
class Modifier:
    def __init__(self, victim_model, predictor):
        self.MASK_ID = 103
        self.predictor = predictor
        self.victim_model = victim_model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_prob(self, model, states):
        with torch.no_grad():
            outputs = model(input_ids=states[0].to(device),
                            token_type_ids=states[1].to(device),
                            attention_mask=states[2].to(device))
        prob = torch.softmax(outputs.logits, -1)
        return prob.to('cpu')

    def __modify__(self, states, actions):
        labels = states[-1]
        lengths = states[3]
        # 保存原来位置上的单词
        original_words = []
        for i, location in enumerate(actions):
            # sentence = self.tokenizer.convert_ids_to_tokens(states[0][i])
            # print(f'sentence: {sentence}')
            # original_words.append(int(states[0][i][location]))
            states[0][i][location] = self.MASK_ID
        predictions = self.get_prob(self.predictor, states)
        for i, location in enumerate(actions):
            # 选取预测的前10个单词，选择概率下降最大的那一个词进行替换
            _, indices = torch.topk(predictions, 10)
            alternations = torch.LongTensor(10, lengths[i])
            for j, index in enumerate(indices[i][location]):
                alternations[j] = states[0][i][:lengths[i]]
                alternations[j][location] = index
            token_type_ids = torch.zeros_like(alternations)
            attention_mask = torch.ones_like(alternations)
            probs = self.get_prob(self.victim_model, [alternations, token_type_ids, attention_mask])
            states[0][i][location] = indices[i][location][torch.argmin(probs, labels[i])[0]]

    def __call__(self, states, actions):
        return self.__modify__(states, actions)


# 利用训练好的模型进行攻击
def attack(dataloader, env, agent):
    # 默认一个句子最多修改的次数
    modify_times = 5
    batch_num = len(dataloader)
    count = 0
    total = 0
    progress = 0
    for states in dataloader:
        states = list(states)
        total += states[0].size()[0]
        for modify_time in range(modify_times):
            # 送入攻击模型获得actions
            actions = agent.get_actions(states)
            # 与环境交互获得下一个状态
            _, _, dones = env.step(states, actions)
            count += torch.sum(dones).item()
            # 判断一个batch里是否所有句子全部攻击成功,如果成功则跳到下一个batch
            if states[0].size()[0] == 0:
                break
        # 展示训练进度
        progress += 1
        if progress % 5 == 0:
            num = math.floor(progress / batch_num * 25)
            progress_bar = '[' + '#' * num + ' ' * (25 - num) + ']'
            print(f'\033[0;31m\r计算攻击成功率：{num * 4}% {progress_bar}\033[0m', end='')
    print(f'\n限制：{modify_times}次修改及以下')
    print(f'攻击成功数：{count}，总数：{total}，攻击成功率：{count/total*100}%')


# 利用强化学习训练攻击模型
def train(dataloader, env, agent, dqn):
    # 默认一个句子最多修改的次数
    modify_times = 10
    # 总共训练的回合数
    epochs = 100
    total_num = len(dataloader)
    for epoch in range(epochs):
        count = 0
        for states in dataloader:
            states = list(states)
            for modify_time in range(modify_times):
                lengths = states[3]
                orginal_states = [states[0][i][:length].clone() for i, length in enumerate(lengths)]
                # 送入攻击模型获得actions
                actions = agent.get_actions(states)
                # 与环境交互获得下一个状态
                observations, rewards, dones = env.step(states, actions)
                next_states = [observations[i][:length] for i, length in enumerate(lengths)]
                # 将数据存入缓冲区
                dqn.store_into_buffer(orginal_states, actions, rewards, next_states, dones)
                # 对Agent进行训练
                dqn.train()
                # 判断一个batch里是否所有句子全部攻击成功,如果成功则跳到下一个batch
                if states[0].size()[0] == 0:
                    break
            # 展示训练进度
            count += 1
            if count % 5 == 0:
                num = math.floor(count / total_num * 25)
                progress_bar = '[' + '#' * num + ' ' * (25 - num) + ']'
                print(f'\033[0;31m\r第{epoch}回合：{num * 4}% {progress_bar}\033[0m', end='')
        print('\n保存模型...')
        agent.attack_model.save_pretrained('AttackModel-v2')


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_set = pd.read_csv('correct_train.tsv', sep='\t')
    # 筛选出标签为negative的数据
    train_set = train_set[train_set['label'] == 0]
    train_dataset = sst2_Dataset(train_set, tokenizer=tokenizer)
    # 加载DataLoader
    TRAIN_BATCH_SIZE = 64
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    # 初始化强化学习的相关类
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    victim_model = BertForSequenceClassification.from_pretrained('VictimBert').to(device)
    attack_model = BertForTokenClassification.from_pretrained('AttackModel', num_labels=1).to(device)
    predictor = BertForMaskedLM.from_pretrained('BertForMaskedLM').to(device)
    env = Env(victim_model, Modifier(victim_model, predictor))
    agent = Agent(attack_model, 0.85)
    dqn = DQN(model=attack_model,
              buffer_size=1024,
              batch_size=64,
              interval=20)
    train(train_dataloader, env, agent, dqn)
    # attack(train_dataloader, env, agent)
