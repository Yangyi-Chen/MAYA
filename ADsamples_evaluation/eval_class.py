import math
import transformers
import language_tool_python
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer, util


class GPT2LM:
    def __init__(self):
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2-large")
        self.lm = transformers.GPT2LMHeadModel.from_pretrained("gpt2-large")
        # self.lm = torch.load('gpt2-large.pkl')

    def __call__(self, sent):
        """
        :param str sent: A sentence.
        :return: Fluency (ppl).
        :rtype: float
        """

        ipt = self.tokenizer(sent, return_tensors="pt", verbose=False)
        return math.exp(self.lm(**ipt, labels=ipt.input_ids)[0])


class GrammarChecker:
    def __init__(self):
        self.lang_tool = language_tool_python.LanguageTool('en-US')

    def check(self, sentence):
        '''
        :param sentence:  a string
        :return:
        '''
        matches = self.lang_tool.check(sentence)
        return len(matches)


class SentenceEncoder:
    def __init__(self, device='cuda'):
        '''
        different version of Universal Sentence Encoder
        https://pypi.org/project/sentence-transformers/
        '''
        self.model = SentenceTransformer('paraphrase-distilroberta-base-v1', device)

    def encode(self, sentences):
        '''
        can modify this code to allow batch sentences input
        :param sentence: a String
        :return:
        '''
        if isinstance(sentences, str):
            sentences = [sentences]

        return self.model.encode(sentences, convert_to_tensor=True)

    def get_sim(self, sentence1, sentence2):
        '''
        can modify this code to allow batch sentences input
        :param sentence1: a String
        :param sentence2: a String
        :return:
        '''
        embeddings = self.model.encode([sentence1, sentence2], convert_to_tensor=True)
        cos_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return cos_sim.item()

    # find adversarial sample in advs which matches ori best
    def find_best_sim(self, ori, advs, find_min=False):
        ori_embedding = self.model.encode(ori, convert_to_tensor=True)
        adv_embeddings = self.model.encode(advs, convert_to_tensor=True)
        best_adv = None
        best_index = None
        best_sim = 10 if find_min else -10
        for i, adv_embedding in enumerate(adv_embeddings):
            sim = util.pytorch_cos_sim(ori_embedding, adv_embedding).item()
            if find_min:
                if sim < best_sim:
                    best_sim = sim
                    best_adv = advs[i]
                    best_index = i

            else:
                if sim > best_sim:
                    best_sim = sim
                    best_adv = advs[i]
                    best_index = i

        return best_adv, best_index, best_sim


class USE:
    def __init__(self):
        self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def count_use(self, sentence1, sentence2):
        embeddings = self.embed([sentence1, sentence2])

        vector1 = tf.reshape(embeddings[0], [512, 1])
        vector2 = tf.reshape(embeddings[1], [512, 1])

        return tf.matmul(vector1, vector2, transpose_a=True).numpy()[0][0]


class Formalizer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def formalize(self, sentence):
        tokens = self.tokenizer.tokenize(sentence)
        return self.tokenizer.convert_tokens_to_string(tokens)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    a = SentenceEncoder()
    b = a.get_sim('you are my mom!', 'I am your mom!')
    print(b)

