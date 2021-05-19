#!/usr/bin/env python
# coding: utf-8


import os, sys
import json
import rntn as nnet
import tree as tree
import pandas as pd
import cPickle as pickle
import numpy as np
import nltk

UNK = 'UNK'
word_map = tree.loadWordMap()

label_names = ["ROOT_SENTENCE", "SYMBOL", "PUNCT", "AND", "OR", "KEY_C", "KEY_NC", "CONDITION", "VARIABLE", "STATEMENT",
               "CAUSE", "EFFECT", "CAUSE_EFFECT_RELATION", "SEPARATEDCAUSE", "INSERTION", "NEGATION", "NONE_CAUSAL",
               "SEPARATEDSTATEMENT", "SEPARATEDAND", "SENTENCE", "SEPARATEDCAUSE_EFFECT_RELATION", "SEPARATEDNEGATION",
               "WORD", "SEPARATEDNONE_CAUSAL", "SEPARATEDOR", "SEPARATEDEFFECT", "SEPARATEDVARIABLE",
               "SEPARATEDCONDITION"]


def load_vocab_index(input_text, df):
    """
    load word map from net and tokenize input sentece with it
    :param input_text:
    :param df:
    :return:
    """
    a = []
    tokenized_sentence = nltk.word_tokenize(input_text)
    for token in tokenized_sentence:
        # we only saved lower case words
        if token.lower() in word_map:
            a.append(word_map[token.lower()])
        else:
            print("Unkown token in the sentence " + token)
            a.append(word_map[UNK])
            row = {"sentence": input_text, "unknown_token": token}
            df = df.append(row, ignore_index = True)

    return a, tokenized_sentence, df


def create_new_token(token_index_1, token_index_2, tokenized_sentence_in_method):
    merged_tokens = tokenized_sentence_in_method[token_index_1] + " " + tokenized_sentence_in_method[token_index_2]
    tokenized_sentence_in_method.append(merged_tokens)
    return len(tokenized_sentence_in_method) - 1, tokenized_sentence_in_method


def perform_evaluation(iterations = 1000):
    df_unknown = pd.DataFrame()
    df_preds = pd.DataFrame()
    RESULTS_DIR = './results/new_results/results_account3/pos_tags/2020_07_23---15_44_766248'

    with open(RESULTS_DIR + "/checkpoint.bin", 'r') as fid:
        opts = pickle.load(fid)
        _ = pickle.load(fid)
        rnn = nnet.RNN(opts.wvecDim, opts.outputDim, opts.numWords, opts.optimizer_settings['minibatch'])
        rnn.initParams()
        rnn.fromFile(fid)
    rnn.L, rnn.V, rnn.W, rnn.b, rnn.Ws, rnn.bs = rnn.stack

    texts = ["If the content is not available in the detected system language, English is selected by default."]

    for text in texts:
        vocab_index, tokenized_sentence_original, df_unknown = load_vocab_index(text, df_unknown)
        word_embeddings = np.zeros((opts.wvecDim, len(vocab_index)))
        for i, index in enumerate(vocab_index):
            word_embeddings[:, i] = rnn.L[:, index]

        probabilities = np.zeros((2, word_embeddings.shape[1]))

        for index in range(word_embeddings.shape[1]):
            # Softmax
            word_embedding = word_embeddings[:, index]
            probs = np.dot(rnn.Ws, word_embedding) + rnn.bs
            probs -= np.max(probs)
            probs = np.exp(probs)
            probs = probs / np.sum(probs)
            probabilities[:, index] = [probs[np.argmax(probs)], np.argmax(probs)]

        start = np.concatenate([word_embeddings, probabilities])
        token_index = np.arange(0, len(tokenized_sentence_original))
        token_index = token_index[:, None].T
        start = np.concatenate([start, token_index])

        # add two dimension two store the indexes of the childs
        cache = np.zeros((start.shape[0] + 2, start.shape[1]))
        # init the total_certainteis dimenison
        working_tensor_empty = np.repeat(start[:, :, np.newaxis], start.shape[1] - 1, axis = 2)

        num_runs = iterations
        tokenized_sentences_array = []
        certainties_runs = []
        total_certainties = np.zeros_like(working_tensor_empty)
        total_certainties = np.repeat(total_certainties[:, :, :, np.newaxis], num_runs, axis = 3)
        for h in range(0, num_runs):
            certainty = 0.0
            tokenized_sentence = list(tokenized_sentence_original)
            # binary tree has maximum height of n-1 (start.shape[1]-1)
            working_tensor = np.repeat(start[:, :, np.newaxis], start.shape[1] - 1, axis = 2)
            # i is the counter for the height of the tree, going through the third dimension of the tensor
            for i in range(0, start.shape[1] - 1):
                # j is the counter for the predictions done in one step
                cache = np.zeros_like(cache)
                for j in range(0, start.shape[1] - 1):
                    k = j + 1
                    # affine with j and j+1
                    if np.isnan(working_tensor[0, j, i]):
                        continue

                    while np.isnan(working_tensor[0, k, i]) and k < start.shape[1] - 1:
                        k = k + 1
                        if not np.isnan(working_tensor[0, k, i]):
                            break

                    lr = np.hstack([working_tensor[:-3, j, i], working_tensor[:-3, k, i]])
                    cache[:-5, k] = np.dot(rnn.W, lr) + rnn.b
                    cache[:-5, k] += np.tensordot(rnn.V, np.outer(lr, lr), axes = ([1, 2], [0, 1]))
                    # Tanh
                    cache[:-5, k] = np.tanh(cache[:-5, k])

                    # Compute softmax
                    probs = np.dot(rnn.Ws, cache[:-5, k]) + rnn.bs
                    probs -= np.max(probs)
                    probs = np.exp(probs)
                    probs = probs / np.sum(probs)
                    # add probs and label to the vector
                    cache[-5:-3, k] = [probs[np.argmax(probs)], np.argmax(probs)]
                    cache[-3, k], tokenized_sentence = create_new_token(working_tensor[-1, j, i].astype(int),
                                                                        working_tensor[-1, k, i].astype(int),
                                                                        tokenized_sentence_in_method = tokenized_sentence)
                    # we store here the indices from the working tensor, to know which two cache items are merged
                    cache[-2, k] = j
                    cache[-1, k] = k


                # select one of the best five predictions
                best_three_preds = cache[-5, :].argsort()[-5:]
                # look at the best five preds and filter if they are below
                filterd_best_preds = []
                for index in best_three_preds:
                    prediction = cache[-5, index]
                    if index == cache.shape[1] - 1 and np.count_nonzero(cache[-5, :]) > 1:
                        continue
                    if prediction > 0.01:
                        filterd_best_preds.append(index)
                if not len(filterd_best_preds) == 0:
                    # use random
                    random_index = np.random.choice(len(filterd_best_preds), 1, replace = False)
                    best_cache_index = filterd_best_preds[random_index[0]]

                best_cache = cache[:, best_cache_index]
                certainty += best_cache[-5]
                # we push the best merge to the next tensor slice
                second_child_of_cache = best_cache[-1].astype(int)
                # We need to propagate/repeat the merged values to every higher dimension
                best_cache_all_dims = np.repeat(best_cache[:-2, np.newaxis], working_tensor.shape[2] - i - 1, axis = 1)

                working_tensor[:, second_child_of_cache, i + 1:] = best_cache_all_dims
                # since we always put the merged result in the second place, set the first place in the upcoming slices to NaN
                # We need to set the value to nan which is the first child node, if there are NaNs in betweeen this does not work
                # We can find the first child by extending the cache with two dimenesions and setting there the index of the childs
                first_child_of_cache = best_cache[-2].astype(int)
                working_tensor[:, first_child_of_cache, i + 1:] = np.nan

            tokenized_sentences_array.append(tokenized_sentence)
            total_certainties[:, :, :, h] = working_tensor
            certainties_runs.append(certainty)

        best_run = np.argmax(certainties_runs)

        best_working_tensor = total_certainties[:, :, :, best_run]

        tensor_slice = best_working_tensor[-2:, :, :]

        test = np.reshape(tensor_slice, (2, -1))

        predictions_dict = []

        for col in test.T:
            label = col[0]
            if not np.isnan(col[1]):
                token123 = tokenized_sentences_array[best_run][col[1].astype(int)]
                if not [token123, label_names[label.astype(int)]] in predictions_dict:
                    predictions_dict.append([token123, label_names[label.astype(int)], text])

        predictions_dict.sort(key = lambda x: len(x[0]), reverse = False)
        df = pd.DataFrame.from_records(predictions_dict, columns = ['token', 'label', 'sentence'])
        df_preds = df_preds.append(df)

    df_preds.to_csv("predictions.csv", header = True, index = False)
    df_unknown.to_csv("unknown_words.csv", header = True, index = False)


if __name__ == '__main__':
    iterations = 20000
    perform_evaluation(iterations = iterations)
