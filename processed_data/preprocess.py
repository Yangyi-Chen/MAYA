import stanza
import pandas as pd
import math


# 将数据集中的单词全部转变为lemma和pos
def transfer():
    nlp = stanza.Pipeline('en', processors='tokenize, mwt, pos, lemma')
    types = ['test']
    for file in types:
        dataset = pd.read_csv(file+'.tsv', sep='\t')
        samples = dataset['sentence'].values
        upos_df = pd.DataFrame(columns=['upos'])
        lemma_df = pd.DataFrame(columns=['lemma'])
        length = len(samples)
        for count, sample in enumerate(samples):
            processed_sentences = nlp(sample)
            words = processed_sentences.sentences[0].words
            upos_str = ' '.join(word.upos for word in words)
            lemma_str = ' '.join(word.lemma for word in words)
            upos_df = upos_df.append({'upos': upos_str}, ignore_index=True)
            lemma_df = lemma_df.append({'lemma': lemma_str}, ignore_index=True)
            # 展示进度
            if count % 10 == 0:
                num = math.floor(count / length * 25)
                progress_bar = '[' + '#' * num + ' ' * (25 - num) + ']'
                print(f'\033[0;31m\r{file}集  转换进度：{num * 4}% {progress_bar}\033[0m', end='')
        # lemma文件附带label
        lemma_df = pd.concat((lemma_df, dataset['label']), axis=1)
        upos_df.to_csv(file + '.upos.tsv', sep='\t', index=False)
        lemma_df.to_csv(file + '.lemma.tsv', sep='\t', index=False)
        print(f'\n\033[0;31m保存至{file}.upos.tsv，{file}.lemma.tsv\033[0m')


if __name__ == '__main__':
    transfer()

