# Evaluation Metrics


## Perplexity 
Implemented in GPTLM.py
```angular2html
lm = GPT2LM()
sent = 'he plants a tree'
ppl = lm(sent)
```


## Grammar Error
Implemented in CheckGrammar.py 
```angular2html
checker = GrammarChecker()
sent = 'he plant a tree'
error_nums = checker.check(sent)
```


## Similarity 
Implemented in Similarity
```angular2html
encoder = SentenceEncoder()

# compute embedding for one setence
embed = encoder.encode('he plants a tree')

# compute similarity of two sentences
sim = encoder.get_sim('he plants a tree', 'she plants a tree')
```