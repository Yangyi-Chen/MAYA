import abc
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from lib.BaiduTransAPI_forPython3 import Translator
from paraphrase_models.style_transfer_paraphrase.inference_utils import GPT2Generator


class Paraphraser(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def paraphrase(self, sentences):
        raise Exception("Abstract method 'substitute' method not be implemented!")


class BackTranslation(Paraphraser):
    def __init__(self):
        super().__init__()
        self.translator = Translator()

    def paraphrase(self, sentences):
        translations = self.translator.translate('en', 'de', sentences)
        if translations is None:
            return None
        back_translations = self.translator.translate('de', 'en', translations)
        if back_translations is None:
            return None
        else:
            return back_translations


class GPT2Paraphraser(Paraphraser):
    def __init__(self, path, device):
        super().__init__()
        self.device = device
        self.paraphraser = GPT2Generator(path, upper_length='same_5', top_p=0.6, device=device)

    def paraphrase(self, sentences):
        with torch.no_grad():
            divide = False
            while True:
                try:
                    if divide:
                        results = []
                        for input in sentences:
                            result, score = self.paraphraser.generate_batch([input])
                            results += result

                        return results

                    else:
                        results, _ = self.paraphraser.generate_batch(sentences)
                        return results

                except Exception as e:
                    if not divide:
                        divide = True
                    else:
                        print('GPU out of memory when getting gpt2 paraphrases!')
                        return None


class T5(Paraphraser):
    def __init__(self, device='cuda'):
        super().__init__()
        model_name = 'tuner007/pegasus_paraphrase'
        self.max_length = 512
        self.device = device
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name,
                                                                     max_length=self.max_length,
                                                                     max_position_embeddings=self.max_length).to(self.device)

    def paraphrase(self, sentences):
        with torch.no_grad():
            tgt_text = []
            for sentence in sentences:
                batch = self.tokenizer.prepare_seq2seq_batch([sentence],
                                                             truncation=True,
                                                             padding='longest',
                                                             max_length=int(len(sentence.split(' '))*1.2),
                                                             return_tensors="pt").to(self.device)

                translated = self.model.generate(**batch,
                                                 max_length=self.max_length,
                                                 min_length=int(len(sentence.split(' '))*0.8),
                                                 num_beams=1,
                                                 num_return_sequences=1,
                                                 temperature=1.5)

                tgt_text += self.tokenizer.batch_decode(translated, skip_special_tokens=True)
            return tgt_text


if __name__ == '__main__':
    def create_next(string):
        string_list = list(string)
        next_list = [0]

        for i in range(1, len(string)):
            last = next_list[i-1]
            if string_list[i] == string_list[last]:
                next_list.append(last+1)

            else:
                while True:
                    last = next_list[last-1]
                    if string_list[i] == string_list[last]:
                        next_list.append(last + 1)
                        break

                    elif last == 0:
                        next_list.append(0)
                        break

        return next_list

    def contain(main, sub):
        main_list = list(main)
        sub_list = list(sub)
        next_list = create_next(sub)

        main_pointer = sub_pointer = 0
        while main_pointer < len(main_list):
            if main_list[main_pointer] == sub_list[sub_pointer]:
                sub_pointer += 1
                main_pointer += 1
                if sub_pointer == len(sub_list):
                    return True

            else:
                if sub_pointer == 0:
                    main_pointer += 1

                else:
                    sub_pointer = next_list[sub_pointer-1]

        return False

    print(contain('acddeff', 'def'))

    pass

"""
 def paraphrase(self, sentences):
        with torch.no_grad():
            divide = False
            while True:
                try:
                    if divide:
                        tgt_text = []
                        for sentence in sentences:
                            batch = self.tokenizer.prepare_seq2seq_batch([sentence],
                                                                         truncation=True,
                                                                         padding='longest',
                                                                         max_length=self.max_length,
                                                                         return_tensors="pt").to(self.device)

                            translated = self.model.generate(**batch,
                                                             max_length=self.max_length,
                                                             min_length=1,
                                                             num_beams=1,
                                                             num_return_sequences=1,
                                                             temperature=1.5)

                            tgt_text += self.tokenizer.batch_decode(translated, skip_special_tokens=True)
                        return tgt_text

                    else:
                        batch = self.tokenizer.prepare_seq2seq_batch(sentences,
                                                                     truncation=True,
                                                                     padding='longest',
                                                                     max_length=self.max_length,
                                                                     return_tensors="pt").to(self.device)

                        translated = self.model.generate(**batch,
                                                         max_length=self.max_length,
                                                         min_length=1,
                                                         num_beams=1,
                                                         num_return_sequences=1,
                                                         temperature=1.5)
                        tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
                        return tgt_text

                except Exception as e:
                    if not divide:
                        divide = True
                    else:
                        print('GPU out of memory when getting t5 paraphrases!')
                        return None
                        """