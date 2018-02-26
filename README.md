# NLU_ASSIGNMENT_1
Assignment-1 for NLU Batch 2018

The repository contains two main files:

1. Perplexity.py
2. Sentence.py

1. Perplexity.py contains the code used to calculate the perplexity of the components of the test set given to the program by mentioning the count , the training corpus and the test corpus.

2. Sentence.py contains the code used to generate a random sentence of length 10 from the training corpus and the test corpus provided by the user, along with the count of the no. of random samples to be taken from the test to generate the sentence (Note: The computation time of the code increases many folds with increase in the no. of random samples).


In both the programs upon execution the user would be asked for preference for the type of corpus to be used i.e Brown or Gutenberg and/or the no. of iterations the program has to run.

Note:
The Kneser Ney smoothing seems to have a multifold complexity thus running it on high no. of iterations will required large resources.
