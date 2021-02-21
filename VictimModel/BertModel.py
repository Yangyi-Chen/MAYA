import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import math


# 情感分析Dataset类
class sst2_Dataset(Dataset):
    def __init__(self, dataset, tokenizer):
        # dataset是用pandas读取的DataFrame类，tokenizer是BertTokenizer
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.len = len(dataset)

    def __getitem__(self, index):
        这个地方对数据的处理应该尽量方法init方法里面，因为这个在跑模型的时候每次都会重复以下这些操作（包括convert token， 创建tensor)，比较费时间，最好在初始化的时候全部做好
        
        sentence, label = train_set.iloc[index, :]
        # 获取句子的tokens
        tokens = ['[CLS]']
        sentence_tokens = self.tokenizer.tokenize(sentence)
        tokens += sentence_tokens + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # 将id和label全部变成tensor类型
        input_ids = torch.tensor(input_ids)
        label = torch.tensor(label)

        token_type_ids = torch.zeros_like(input_ids)
        return input_ids, token_type_ids, label

    def __len__(self):
        return self.len


# DataLoader类的函数参数
def collate_fn(data):
    input_ids = [single_data[0] for single_data in data]
    token_type_ids = [single_data[1] for single_data in data]
    label = [single_data[2] for single_data in data]
    # 利用zero_padding填充不等长的句子
    input_ids = pad_sequence(input_ids, batch_first=True)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True)
    attention_mask = torch.ones_like(token_type_ids)   这里的attention mask 应该有点问题，最好再查看一下原来博客是怎么做的 https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html
    return input_ids, token_type_ids, attention_mask, label


# 训练模型
def train(net, trainloader, testloader=None):
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)   最好用 AdamW， weight_decay 当做超参数进行调整
    epoch = 20
    total_num = len(trainloader)
    max_accuracy = 0
    for i in range(epoch):
        count = 0
        net.train()
        for data in trainloader:
            optimizer.zero_grad()
            data = [torch.tensor(column).to(device) for column in data]
            outputs = net(input_ids=data[0],
                          token_type_ids=data[1],
                          attention_mask=data[2],
                          labels=data[3])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            count += 1
            # 展示训练进度
            if count % 5 == 0:
                num = math.floor(count / total_num * 25)
                progress_bar = '[' + '#' * num + ' ' * (25 - num) + ']'
                print(f'\033[0;31m\r第{i}回合：{num * 4}% {progress_bar}\033[0m', end='')
        print('\n')
        # 一个回合后检查正确率
        print(f'第{i}回合训练集')
        get_prediction(net, trainloader)
        if testloader is not None:
            print(f'第{i}回合测试集')
            accuracy = get_prediction(net, testloader)
            # 选择测试集上正确率最高的模型进行保存
            if max_accuracy < accuracy:
                max_accuracy = accuracy
                net.save_pretrained('VictimBert')


# 计算模型正确率
def get_prediction(net, dataloader):
    net.eval()
    batch_size = dataloader.batch_size
    total = 0
    correct = 0
    with torch.no_grad():
        for data in dataloader:
            data = [torch.tensor(column).to(device) for column in data]
            outputs = net(input_ids=data[0],
                          token_type_ids=data[1],
                          attention_mask=data[2])
            prediction = outputs.logits
            _, prediction = torch.max(prediction, 1)
            correct += (prediction == data[3]).sum().item()
            total += batch_size   这里直接+batch_size 应该有点问题， 最后一个batch 应该不足 batch_size 大小
    print('正确率为{:.2f}%'.format(correct / total * 100))
    return correct / total * 100


if __name__ == '__main__':
    # 加载BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # 加载Dataset
    train_set = pd.read_csv('train.tsv', sep='\t')
    test_set = pd.read_csv('test.tsv', sep='\t')
    train_dataset = sst2_Dataset(train_set, tokenizer=tokenizer)
    test_dataset = sst2_Dataset(test_set, tokenizer=tokenizer)
    # 加载DataLoader
    TRAIN_BATCH_SIZE = 128
    TEST_BATCH_SIZE = 128
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, collate_fn=collate_fn)  这里要shuffle
    test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, collate_fn=collate_fn)
    # 加载已有的情感分类模型BertForSequenceClassification
    net = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    net = net.to(device)
    train(net, train_dataloader, test_dataloader)
