
from sentence_transformers import SentenceTransformer, util
class SentenceEncoder():
    def __init__(self):
        '''
        different version of Universal Sentence Encoder
        https://pypi.org/project/sentence-transformers/
        '''
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')

    def encode(self, sentence):
        '''
        can modify this code to allow batch sentences input
        :param sentence: a String
        :return:
        '''
        embedding = self.model.encode([sentence], convert_to_tensor=True)
        return embedding

    def get_sim(self, sentence1, sentence2):
        '''
        can modify this code to allow batch sentences input
        :param sentence1: a String
        :param sentence2: a String
        :return:
        '''
        sent1_embed = self.model.encode([sentence1], convert_to_tensor=True)
        sent2_embed = self.model.encode([sentence2], convert_to_tensor=True)
        cos_sim = util.pytorch_cos_sim(sent1_embed, sent2_embed)
        return cos_sim



if __name__ == '__main__':
    encoder = SentenceEncoder()
    a = encoder.encode('he plants a tree').squeeze()
    sim = encoder.get_sim('he plants a tree', 'she plants a tree').squeeze()