import pandas as pd
from supar import Parser
from nltk.tree import Tree


class ConstituencyParser:
    def __init__(self):
        self.parser = Parser.load('crf-con-en')

    @staticmethod
    def __sentence_to_list(sentence):
        word_list = sentence.strip().replace('(', '[').replace(')', ']').split(' ')

        while '' in word_list:
            word_list.remove('')

        return word_list

    def get_tree(self, sentence):
        word_list = self.__sentence_to_list(sentence)
        if len(word_list) == 0:
            return None

        try:
            prediction = self.parser.predict(word_list, verbose=False)
            return prediction.trees[0]

        except Exception as e:
            print('error: cannot get tree!')
            return None

    def __call__(self, sentence):
        # 返回句子的每个节点及索引
        root = self.get_tree(sentence)
        if root is None:
            return None, []

        node_list = pd.DataFrame(columns=['sub_tree', 'phrase', 'index', 'label', 'length'])

        for index in root.treepositions():
            sub_tree = root[index]
            if isinstance(sub_tree, Tree):
                if len(sub_tree.leaves()) > 1:
                    phrase = ' '.join(word for word in sub_tree.leaves())
                    node_list = node_list.append({'sub_tree': sub_tree,
                                                  'phrase': phrase,
                                                  'index': index,
                                                  'label': sub_tree.label(),
                                                  'length': len(sub_tree.leaves())}, ignore_index=True)

        node_list = node_list.drop_duplicates('phrase', keep='last')

        return root, node_list.values
