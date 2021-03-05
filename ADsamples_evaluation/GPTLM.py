import math
import transformers
import torch
class GPT2LM:
    def __init__(self):
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        self.lm = transformers.GPT2LMHeadModel.from_pretrained("gpt2-large.pkl")
        # self.lm = torch.load('gpt2-large.pkl')




    def __call__(self, sent):
        """
        :param str sent: A sentence.
        :return: Fluency (ppl).
        :rtype: float
        """

        ipt = self.tokenizer(sent, return_tensors="pt", verbose=False)
        return math.exp(self.lm(**ipt, labels=ipt.input_ids)[0])


if __name__ == '__main__':
    lm = GPT2LM()
    print(lm('He plants a tree.'))