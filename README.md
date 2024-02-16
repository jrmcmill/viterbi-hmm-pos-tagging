# Part of Speech Tagging

## via an HMM following the Viterbi algorithm

This is my naive approach to solving the POS tagging problem using a hidden Markov model with bigrams using only NumPy.

The annotated dataset I use for training follows the format:

Each line in the file is a sentence such that each word takes the form of "raw_text/TAG"

For example, "mortgage/NN" and "payments/NNS" are a noun (NN) and plural noun (NNS), respectively. There are also symbol tags for punctuation, like ",/," for commas.

For handling unseen words, Laplace smoothing is employed.
