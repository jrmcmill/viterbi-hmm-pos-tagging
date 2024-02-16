"""
Handle data extracting and HMM train/test computations
"""
import numpy as np  # storing intermediate tag score states
import pandas as pd  # exporting test-train results


PROPERS = ['NP', 'NPS']  # for correctly standardizing to lower


class ViterbiPOS:
    """
    Implementation of the Viterbi algorithm for Part-of-Speech (POS) tagging.

    Attributes:
        train_tag_frequencies (dict): Dictionary containing counts of each POS tag in the training data.
        known_words (set): Set of unique words seen in the training data.
        train_bigram_counts (dict): Dictionary with counts of each POS tag bigram in the training data.
        train_bigram_probabilities (dict): Dictionary with conditional probabilities of each POS tag bigram.
        train_lexical_probabilities (dict): Dictionary with conditional probabilities of each word given a POS tag.
        train_tags (list): List of POS tags present in the training data.
        word_count (int): Total count of words in the training data.
        trained (bool): Flag indicating whether the model has been trained.

    Methods:
        get_tag_frequencies(): Extracts POS tag frequencies and seen words from the training data.
        get_bigram_counts(): Computes POS tag bigram counts from the training data.
        get_bigram_probabilities(b_counts, t_counts): Computes conditional probabilities of POS tag bigrams.
        get_lexical_probabilities(t_counts): Computes conditional probabilities of words given POS tags.
        train(train_filename): Trains the ViterbiPOS model using the specified training data file.
        test(test_filename): Applies the Viterbi algorithm to POS tag a test data file and evaluates accuracy.

    Note:
        Before testing, the model must be trained using the `train` method.
    """
    def __init__(self) -> None:
        """
        Initializes a new instance of the ViterbiPOS class.

        Attributes:
            train_tag_frequencies (dict): Dictionary for counting POS tag frequencies.
            known_words (set): Set to store unique words encountered in the training data.
            train_bigram_counts (dict): Dictionary for counting POS tag bigrams.
            train_bigram_probabilities (dict): Dictionary for storing conditional probabilities of POS tag bigrams.
            train_lexical_probabilities (dict): Dictionary for storing conditional probabilities of words given POS tags.
            train_tags (list): List of POS tags present in the training data.
            word_count (int): Total count of words in the training data.
            trained (bool): Flag indicating whether the model has been trained. Default is False.
            train_filename (str): Filename of the training data. Set during training.
        """
        self.train_tag_frequencies = None
        self.known_words = None  # vocabulary
        self.train_bigram_counts = None
        self.train_bigram_probabilities = None
        self.train_lexical_probabilities = None
        self.train_tags = None
        self.word_count = None
        self.trained = False  # don't allow testing until trained
    
    def get_tag_frequencies(self) -> tuple[dict[str: int], set[str]]:
        """
        Extracts POS tag frequencies and seen words from the training data.

        Returns:
            tuple: A tuple containing two elements:
                - A dictionary (tag_frequencies) with POS tags as keys and their respective counts as values.
                - A set (seen_words) containing unique words seen in the training data.

        Note:
            The method assumes that the training data file (specified during training) is already set.
        """
        # BOS = beginning of sentence
        tag_frequencies = {'BOS': 0}  # initialize dictionary and BOS
        seen_words = set()  # initialize set for all seen words

        with open(self.train_filename, 'r') as raw:
            for line in raw:  # for sentence in file
                tag_frequencies['BOS'] += 1  # log BOS and

                for pair in line.split():  # for word in sentence
                    pair = pair.split('/')  # log its tag
                    
                    if len(pair) > 1:  # skips irregular formatting
                        if pair[-1] not in tag_frequencies.keys():
                            tag_frequencies[pair[-1]] = 1
                        else:
                            tag_frequencies[pair[-1]] += 1
                        if len(pair) > 2:
                            word = ' '.join(pair[:-1])
                        else:
                            word = pair[0]
                        if pair[-1] not in PROPERS:  # standardize to lower
                            word = word.lower()  # if not proper noun
                        seen_words.add(word)
                    else:
                        continue
        
        # return all tag counts and seen words
        return tag_frequencies, seen_words
    
    def get_bigram_counts(self) -> dict[tuple[str, str]: int]:
        """
        Computes POS tag bigram counts from the training data.

        Returns:
            dict: A dictionary (bigram_counts) with tuples of adjacent POS tags as keys and their counts as values.

        Note:
            The method assumes that the training data file (specified during training) is already set.
        """
        bigram_counts = {}  # initialize dictionary for counting
        tag_list = []  # initialize empty word list

        with open(self.train_filename, 'r') as raw:
            for line in raw:  # for sentence in file
                tag_list.append('BOS')  # log BOS and

                for pair in line.split():  # for word in sentence
                    if len(pair.split('/')) > 1:
                        tag_list.append(pair.split('/')[-1])  # log its tag
                    else:
                        continue

            tag_list2 = tag_list[1:]  # create copy of tags less first

            for bigram in zip(tag_list, tag_list2):  # and zip for bigrams
                if bigram[1] != 'BOS':  # disregard bigrams across sentences
                    if bigram not in bigram_counts.keys():  # log tag bigram
                        bigram_counts[bigram] = 1
                    else:
                        bigram_counts[bigram] += 1

        return bigram_counts  # return all tag bigram counts
    
    @staticmethod
    def get_bigram_probabilities(b_counts: dict[tuple[str, str]: int]=None, t_counts: dict[str: int]=None) -> dict[tuple[str, str]: float]:
        """
        Computes conditional probabilities of POS tag bigrams.

        Args:
            b_counts (dict): Dictionary containing counts of POS tag bigrams.
            t_counts (dict): Dictionary containing counts of individual POS tags.

        Returns:
            dict: A dictionary (bigram_probabilities) with tuples of adjacent POS tags as keys
                and their conditional probabilities as values.

        Note:
            - The method assumes that both b_counts and t_counts are provided.
            - Probability computation follows the formula P(Tj | Ti) = count(Ti, Tj) / count(Ti).
        """
        bigram_probabilities = {}  # initialize dictionary for counting

        for bigram in b_counts.keys():  # for bigram in counts
            # log P(Tj | Ti)
            bigram_probabilities[bigram] =\
                b_counts[bigram] / t_counts[bigram[0]]  # bc / tc

        return bigram_probabilities  # return all bigram probabilities
    
    # takes filename and tag counts, returns all w given t probabilities
    def get_lexical_probabilities(self, t_counts: dict[str: int]=None) -> dict[tuple[str, str]: float]:
        """
        Computes conditional probabilities of words given POS tags.

        Args:
            t_counts (dict): Dictionary containing counts of individual POS tags.

        Returns:
            dict: A dictionary (lexical_probabilities) with tuples of words and POS tags as keys
                and their conditional probabilities as values.

        Note:
            - The method assumes that the training data file (specified during training) is already set.
            - Probability computation follows the formula P(W | T) = count(W, T) / count(T).
        """
        word_tag_counts = {}  # initialize dictionaries for counting and
        lexical_probabilities = {}  # probabilities

        with open(self.train_filename, 'r') as raw:
            for line in raw:  # for sentence in file
                for pair in line.split():  # for word in sentence
                    pair = pair.split('/')  # generate key and log

                    if len(pair) > 1:
                        if len(pair) > 2:
                            word = ' '.join(pair[:-1])
                        else:
                            word = pair[0]
                        if pair[-1] not in PROPERS:  # if not proper noun
                            word = word.lower()  # standardize lowercase
                        pair = tuple([word, pair[-1]])

                        if pair not in word_tag_counts.keys():
                            word_tag_counts[pair] = 1
                        else:
                            word_tag_counts[pair] += 1
                    else:
                        continue

            for word_tag in word_tag_counts.keys():  # for word, tag
                try:
                    lexical_probabilities[word_tag] = word_tag_counts[word_tag]\
                                                    / t_counts[word_tag[1]]
                except IndexError:
                    continue

        return lexical_probabilities  # return all conditional probabilities
    
    def train(self, train_filename: str=None) -> None:
        """
        Trains the ViterbiPOS model using the specified training data file.

        Args:
            train_filename (str): The filename of the training data.

        Returns:
            None

        Note:
            - The method initializes various attributes of the ViterbiPOS instance based on the training data.
            - The training process includes calculating tag frequencies, bigram counts, bigram probabilities,
            lexical probabilities, and other relevant statistics.
            - The trained model is required for testing, and the method sets the 'trained' attribute to True.
        """
        self.train_filename = train_filename

        # T: count
        self.train_tag_frequencies, self.known = self.get_tag_frequencies()

        # (Ti, Ti+1): count
        self.train_bigram_counts = self.get_bigram_counts()

        # (Ti, Ti+1): conditional probability (Ti+1 | Ti)
        self.train_bigram_probabilities = self.get_bigram_probabilities(self.train_bigram_counts, self.train_tag_frequencies)

        # (W, T): conditional probability (W | T)
        self.train_lexical_probabilities = self.get_lexical_probabilities(self.train_tag_frequencies)

        self.train_tags = list(self.train_tag_frequencies.keys())
        self.train_tags.remove('BOS')  # unneeded beginning of sentence tag

        self.word_count = sum(self.train_tag_frequencies.values())
        self.trained = True

        return
    
    def test(self, test_filename: str=None) -> None:
        """
        Applies the Viterbi algorithm to POS tag a test data file and evaluates accuracy.

        Args:
            test_filename (str): The filename of the test data.

        Returns:
            None

        Note:
            - The method assumes that the model has been trained using the 'train' method.
            - The accuracy is calculated by comparing the model's predicted tags with the ground truth tags.
            - The results, including words, model tags, and true tags, are saved to a CSV file ('results.csv').
        """
        if not self.trained:  # abort test if not yet trained
            print('error: not yet trained')

            return
        
        ground_truth_tags = []  # initialize list for storing ground truth
        model_tags = []  # initialize list for storing model tags
        all_words = []  # initialize for use in accuracy

        with open(test_filename, 'r') as raw:
            # strip tags and store for accuracy
            for line in raw:  # for sentence in file
                clean_line = []  # initialize line for stripping tags

                for pair in line.split():  # for word in sentence
                    pair = pair.split('/')
                    
                    if len(pair) > 2:  # add word to clean line
                        word = ' '.join(pair[:-1])
                    else:
                        word = pair[0]
                    
                    if pair[-1] not in PROPERS:
                        word = word.lower()  # standardize lowercase
                    
                    clean_line.append(word)
                    ground_truth_tags.append(pair[-1])  # store ground truth tag

                # Viterbi begins here
                # initialize array containing words' tag scores
                word_scores = np.zeros([len(self.train_tags), len(clean_line)])

                # for every tag calculate and store first word scores
                for i, tag in zip(range(0, len(self.train_tags)), self.train_tags):
                    try:  # only store score if key exists
                        word_scores[i, 0] = self.train_lexical_probabilities[(clean_line[0], tag)] *\
                                            self.train_bigram_probabilities[('BOS', tag)]
                    except KeyError:  # else skip (leave as 0)
                        continue

                # initialize bp storage to -1 index
                back_pointers = np.empty([len(self.train_tags), len(clean_line)])
                back_pointers.fill(-1)
                # the bp will point to the prior row (tag) index

                # for remaining words in the stripped sentence
                # with column index for array
                for j, word in zip(range(1, len(clean_line)), clean_line[1:]):
                    # for tag in all tags with row index for array
                    for i, tag in zip(range(0, len(self.train_tags)), self.train_tags):
                        try:  # only perform calculation if lex prob exists
                            lex_prob = self.train_lexical_probabilities[(word, tag)]  # store lex prob
                            max_list = []  # initialize max list for calculation

                            # for tag in all tags (max calculation)
                            for k, max_tag in zip(range(0, len(self.train_tags)), self.train_tags):
                                # append prior score times b prob if prob exists
                                try:
                                    max_list.append(word_scores[k, j - 1] *
                                                    self.train_bigram_probabilities[(max_tag, tag)])
                                except KeyError:  # else append 0
                                    max_list.append(0)

                            # set score based on max list and word indices
                            word_scores[i, j] = lex_prob * max(max_list)
                            # set bp based on tag and word indices
                            back_pointers[i, j] = max_list.index(max(max_list))

                        except KeyError:  # else treat as unseen word
                            if word not in self.known:
                                # using laplace smoothing
                                lex_prob = 1 / self.word_count  # store unseen lex prob
                            else:
                                lex_prob = 0  # since word exists without tag

                                # initialize max list for calculation
                                max_list = []

                                # for tag in all tags (max calculation)
                                for k, max_tag in zip(range(0, len(self.train_tags)), self.train_tags):
                                    # append prior score times b prob if exists
                                    try:
                                        max_list.append(word_scores[k, j - 1] *
                                                        self.train_bigram_probabilities[(max_tag, tag)])
                                    except KeyError:  # else append 0
                                        max_list.append(0)

                                # set score based on max list and word indices
                                word_scores[i, j] = lex_prob
                                # set bp based on tag and word indices
                                back_pointers[i, j] = max_list.index(
                                    max(max_list))

                                continue

                            max_list = []  # initialize max list for calculation

                            # for tag in all tags (max calculation)
                            for k, max_tag in zip(range(0, len(self.train_tags)), self.train_tags):
                                # append prior score times b prob if prob exists
                                try:
                                    max_list.append(word_scores[k, j - 1] *
                                                    self.train_bigram_probabilities[(max_tag, tag)])
                                except KeyError:  # else append 0
                                    max_list.append(0)

                            # set score based on max list and word indices
                            word_scores[i, j] = lex_prob * max(max_list)
                            # set bp based on tag and word indices
                            back_pointers[i, j] = max_list.index(max(max_list))

                # get final word's tag with max of its scores
                next_index = list(word_scores[:, -1]).\
                    index(max(word_scores[:, -1]))
                # for following bp tags via index
                indexed_tags = dict(zip(range(0, len(self.train_tags)), self.train_tags))
                # initialize tagging sequence with final
                sequence = [indexed_tags[next_index]]

                for j in reversed(range(1, len(clean_line))):
                    sequence.append(indexed_tags[back_pointers[next_index,
                                                            j]])
                    next_index = int(back_pointers[next_index, j])

                sequence.reverse()  # get correct sequence order
                model_tags.extend(sequence)  # append to model results
                all_words.extend(clean_line)  # append for output file

        correct = 0  # initialize accuracy numerator
        for i, tag in zip(range(0, len(model_tags)), model_tags):
            if tag == ground_truth_tags[i]:
                correct += 1
        print(f'accuracy: {(correct / len(model_tags)) * 100}%')

        # export results to CSV
        out = pd.DataFrame(list(zip(all_words, model_tags,
                                    ground_truth_tags)),
                        columns=['words', 'model_tags', 'true_tags'])
        out.to_csv('results.csv', index=False)

        return

