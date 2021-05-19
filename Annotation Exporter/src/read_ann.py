import argparse
import os
import pandas as pd
import nltk
import re

ROOT_SENTENCE = "ROOT_SENTENCE"
SENTENCE = "SENTENCE"
CAUSE_EFFECT_RELATION = "CAUSE_EFFECT_RELATION"
CAUSE = "CAUSE"
KEY_C = "KEY_C"
STATEMENT = "STATEMENT"
VARIABLE = "VARIABLE"
CONDITION = "CONDITION"
PUNCT = "PUNCT"
COLON = "COLON"
APOSTROPH = "APOSTROPH"
OR = "OR"
AND = "AND"
SYMBOL = "SYMBOL"
SEPERATOR = "SEPERATOR"
NEGATION = "NEGATION"
WORD = "WORD"
PARENTHESES = "PARENTHESES"
Text = "Text"
Label = "Label"
Begin = "Begin"
End = "End"
COMPARATOR = "COMPARATOR"

WordCount = "WordCount"
OffsetCount = "OffsetCount"


def import_ann_file(path):
    file = pd.read_csv(path, header = None, sep = '\n|\t', index_col = 0, names = ['T', 'Type', 'Text'])
    file = file.join(file['Type'].str.split(' ', expand = True).rename(columns = {0: 'Label', 1: 'Begin', 2: 'End'}))
    file = file.drop(['Type'], axis = 1)

    return file


def sort_annotations(data):
    print("Sort annotations method:")
    data['Begin'] = pd.to_numeric(data[Begin], errors = 'coerce')
    data['End'] = pd.to_numeric(data[End], errors = 'coerce')

    return data.sort_values(by = [Begin, End], ascending = [True, False])


def create_string(label, begin, end):
    return label


def has_two_childs(index_of_row, data):
    """

    :param index_of_row: specify parent
    :param data: current dataframe
    :return: the direct children of the caller, with indices indicating the location in the sentence
    """
    parent_row = data.loc[index_of_row]
    begin = parent_row[Begin]
    end = parent_row[End]
    child_count = 0
    mask = (data[Begin] >= begin) & (data[End] <= end)
    children = data[mask]
    list = []

    children = children.sort_values(by = [Begin, WordCount], ascending = [True, False])
    children.drop(children.head(1).index, inplace = True)
    # First element is the element itself

    indexes = children.index.values.tolist()
    # Start with second element and
    for index in indexes:
        row = children.loc[index]
        if begin == end:
            break
        if begin <= row[Begin]:
            begin = row[End]
            child_count += 1
            list.append((row[Text], row[Begin], row[End], row[Label], row[WordCount]))

    return list, parent_row


def root_has_two_childs(df, length, fulltext_offset):
    """

    :rtype: returns true if the root sentence has two children
    """
    indexes = df.index.values.tolist()
    row = df.head(1)
    for index in indexes:
        row = df.loc[index]
        if row[Begin] == (0 + fulltext_offset) and row[End] == (length + fulltext_offset):
            # we have a label which spans from begin to second last character in sentence
            # we can merge this label together with the PUNCT label to ROOT_SENTENCE
            return True

    # We don't have a label which spans from begin to second last character
    return False


def findall(p, s):
    """Yields all the positions of
    the pattern p in the string s."""
    i = s.find(p)
    while i != -1:
        yield i
        i = s.find(p, i + 1)


def create_single_token_row(fulltext, token, label, word_count_df, fulltext_offset, tokens,
                            merged_tokens_indices):
    """
    Creates a single row in the dataframe with a respective label
    :param fulltext: The current sentence as string
    :param token: the token we want to add a label to
    :param label: the label of the token
    :param word_count_df: the dataframe
    :param fulltext_offset: offset including brackets after labeling tokens
    :param tokens: single tokens of sentence
    :param merged_tokens_indices: indices of already merged tokens
    :return:
    """
    # find all occurrences (begin index) of the respective token
    beginnings = [(i, fulltext[i:len(token) + i]) for i in findall(token, fulltext)]

    updated_beginnings = []
    for (begin, text) in beginnings:
        print(begin, text)
        if (label != WORD and (not (token == "," and (re.match(r'[0-9]+', fulltext[begin - 1]) is not None) and (re.match(r'[0-9]+', fulltext[begin + len(token)]) is not None)))) or (
                ((re.match(r"[^A-Za-z0-9.\]/]+", fulltext[begin - 1]) is not None) or begin == 0) and (
                re.match(r'[^A-Za-z0-9/]+', fulltext[
                    begin + len(token)]) is not None)):
            if fulltext[begin - 1] == "-" and (fulltext[begin + len(token)] == "-" or fulltext[begin + len(token)] == "‐"):
                # we have something like left-to-right and don't want to match to
                if (re.match(r'[A-Za-z0-9]+', fulltext[begin - 2]) is not None) and \
                        (re.match(r'[A-Za-z0-9]+', fulltext[begin + len(token) + 1]) is not None):
                    # We check against the characters next to the - (e.g. left-to-right -> T in left and R in Right
                    continue
            if fulltext[begin - 1] == "-" or fulltext[begin - 1] == "‐":
                # we have something like sub-parameter and don't want to match parameter
                if (re.match(r'[A-Za-z0-9]+', fulltext[begin - 2]) is not None):
                    # We check against the characters next to the - (e.g. sub-parameter --> B in sub
                    # We continue if we have a number or text next to the dash
                    continue
            if fulltext[begin + len(token)] == "-" or fulltext[begin + len(token)] == "‐":
                # we have something like sub-parameter and don't want to match sub
                if (re.match(r'[A-Za-z0-9]+', fulltext[begin + len(token) + 1]) is not None):
                    # We check against the characters next to the - (e.g. sub-parameter --> P in parameter
                    # We continue if we have a number or text next to the dash
                    continue
            if fulltext[begin + len(token)] == "_":
                # we have something like sub-parameter and don't want to match sub
                if (re.match(r'[A-Za-z0-9]+', fulltext[begin + len(token) + 1]) is not None):
                    # We check against the characters next to the - (e.g. sub-parameter --> P in parameter
                    # We continue if we have a number or text next to the dash
                    continue
            if fulltext[begin - 1] == "_":
                # we have something like sub-parameter and don't want to match parameter
                if (re.match(r'[A-Za-z0-9]+', fulltext[begin - 2]) is not None):
                    # We check against the characters next to the - (e.g. sub-parameter --> B in sub
                    # We continue if we have a number or text next to the dash
                    continue
            if (re.match(r'[0-9]+', token) is not None) and fulltext[begin - 1] == ",":
                continue
            if (re.match(r'[0-9]+', token) is not None) and fulltext[begin + len(token)] == ".":
                if len(fulltext) > begin+len(token)+1:
                    if re.match(r'[0-9]+', fulltext[begin + len(token) + 1]) is not None:
                        continue
                else:
                    # we have a number at the end
                    if len(merged_tokens_indices) == 0:
                        updated_beginnings.append(begin)
                    else:
                        for a in merged_tokens_indices:
                            print(f"merged token indices: {merged_tokens_indices}")
                            print(f"token in merged token indices: {tokens[a]}")
                            print(f"text to compare: {fulltext[begin:begin + len(token) + 2]}")
                            # +2 for 's und +3 for n't or not
                            if tokens[a] == fulltext[begin:begin + len(token) + 2] or tokens[a] == fulltext[
                                                                                                   begin:begin + len(
                                                                                                       token) + 3]:
                                # The text we matched to the token is there twice for example (peer's and peer)
                                # We don't want to create a row for the peer match of the word peer's
                                continue
                            else:
                                updated_beginnings.append(begin)
            else:
                if len(merged_tokens_indices) == 0:
                    updated_beginnings.append(begin)
                else:
                    for a in merged_tokens_indices:
                        # +2 for 's und +3 for n't or not
                        if tokens[a] == fulltext[begin:begin + len(token) + 2] or tokens[a] == fulltext[
                                                                                               begin:begin + len(
                                                                                                       token) + 3]:
                            # The text we matched to the token is there twice for example (peer's and peer)
                            # We don't want to create a row for the peer match of the word peer's
                            continue
                        else:
                            updated_beginnings.append(begin)

    for new_begin in updated_beginnings:
        row = {Text: token, Label: label, Begin: new_begin + fulltext_offset,
               End: new_begin + len(token) + fulltext_offset,
               WordCount: 1,
               OffsetCount: len(label) + 3}
        word_count_df = word_count_df.append(row, ignore_index = True)

    return word_count_df


def simple_merge(children, df, parent_row, fulltext, fulltext_offset):
    """
    Takes the children array (which contains only the first level children) and merges them together
    Merge everything between start and end, starting from left
    We only care about the fist level children, since we call this function on each pair of words
    :param children: array with children to merge
    :param df: dataframe
    :return:
    """
    cache = []
    for index, child in enumerate(reversed(children)):
        if index == len(children) - 1:
            # We reached the end
            continue

        # start of merge
        if index == 0:
            cache = child
            continue

        if parent_row[Label] == NEGATION and child[3] == WORD:
            label = NEGATION
        elif parent_row[Label] == PARENTHESES and child[3] == WORD:
            label = PARENTHESES
        elif parent_row[Label] == CONDITION and child[3] == WORD:
            label = CONDITION
        elif parent_row[Label] == VARIABLE and child[3] == WORD:
            label = VARIABLE
        elif parent_row[Label] == STATEMENT and child[3] == WORD:
            label = STATEMENT
        elif parent_row[Label] == CAUSE_EFFECT_RELATION and child[0] == ",":
            label = "SEPARATED" + cache[3]
        else:
            label = parent_row[Label]

        if fulltext[cache[1] - 1 - fulltext_offset] == " ":
            text = child[0] + " " + cache[0]
        else:
            text = child[0] + cache[0]
        begin = child[1]
        end = cache[2]
        word_count = cache[4] + child[4]

        row = {Text: text, Label: label, Begin: begin, End: end,
               WordCount: word_count,
               OffsetCount: len(label) + 3}
        cache = (row[Text], row[Begin], row[End], row[Label], row[WordCount])

        df = df.append(row, ignore_index = True)

    print("This is the dataframe after merging")
    print(df.to_string())
    return df


def separator_merge(start, end, children, df, fulltext, fulltext_offset, label, entire_statement=False):
    """
    Takes the children array (which contains only the first level children) and merges them together
    First merge everything between start and end, then merge this combined thing with the end and then this together with the start
    We only care about the fist level children, since we call this function on each pair of words

    :param index:
    :param start: startindex of children array
    :param end: endindex of children array
    :param children: array with children to merge
    :param df: dataframe
    :return:
    """
    origin_end = end
    cache = []
    origin_label = label
    print(f"This is the children length {len(children[start:end + 1])} of the children array {children[start:end + 1]}")
    for index, child in enumerate(reversed(children[start:end + 1])):
        if index == end:
            # We reached the end merge with first element and start
            continue
        else:
            label = origin_label

        # begin
        if index == 0:
            # leave the first child for the last merge
            continue
        if index == 1:
            cache = child
            continue
        if fulltext[cache[1] - 1 - fulltext_offset] == " ":
            text = child[0] + " " + cache[0]
        else:
            text = child[0] + cache[0]
        begin = child[1]
        end = cache[2]
        word_count = cache[4] + child[4]

        row = {Text: text, Label: label, Begin: begin, End: end,
               WordCount: word_count,
               OffsetCount: len(label) + 3}
        cache = (row[Text], row[Begin], row[End], row[Label], row[WordCount])

        df = df.append(row, ignore_index = True)

    if not entire_statement:
        # we have merged everything except the first element, if it is not an entire statement merge the first separator
        # with the cache
        if fulltext[children[origin_end][2] - 1 - fulltext_offset] == " ":
            text = cache[0] + " " + children[origin_end][0]
        else:
            text = cache[0] + children[origin_end][0]
        label = "INSERTION"
        begin = cache[1]
        end = children[origin_end][2]
        word_count = cache[4] + children[origin_end][4]

        row = {Text: text, Label: label, Begin: begin, End: end,
               WordCount: word_count,
               OffsetCount: len(label) + 3}
        df = df.append(row, ignore_index = True)
        # replace all children, which are now merged with the merged row, to continue with the simple merge
        children[start:end + 1] = [(text, begin, end, label, word_count)]

    print("After separator merge")
    print(children)
    print(df.to_string())
    print("End separator merge")

    return df, children


def check_if_in_single_labels(tokens, single_labels, single_labels_extended, fulltext, fulltext_offset, df):
    symbols = {'``', '"', '”', '>', "''", "“", "”", "(", ")", "-", "‐", ":", ";", "’", "[", "]", "{", "}", "&", "%", "$",
               "§"}
    punct = {",", "."}
    for token in tokens:
        # we only look at the labels we remove after this method call
        if token in single_labels:
            # get all the occurences of the respective token
            beginnings = [(i, fulltext[i:len(token) + i]) for i in findall(token, fulltext)]

            for (begin, text) in beginnings:
                for single_labels_text, single_labels_begin in single_labels_extended:
                    if text == single_labels_text and begin != (single_labels_begin - fulltext_offset) \
                            and (text, begin + fulltext_offset) not in single_labels_extended:
                        # we found a token in the fulltext, which has the same name but no single_label assigned
                        # asssign a label to it
                        if (((re.match(r'[^A-Za-z0-9.\]]+', fulltext[begin - 1]) is not None) or begin == 0) and (
                                re.match(r'[^A-Za-z0-9]+', fulltext[
                                    begin + len(token)]) is not None)):
                            # Check if this is really a word
                            if text in symbols:
                                single_row = {Text: token, Label: SYMBOL, Begin: begin + fulltext_offset,
                                              End: begin + len(token) + fulltext_offset,
                                              WordCount: 1,
                                              OffsetCount: len(SYMBOL) + 3}
                            elif text in punct:
                                single_row = {Text: token, Label: PUNCT, Begin: begin + fulltext_offset,
                                              End: begin + len(token) + fulltext_offset,
                                              WordCount: 1,
                                              OffsetCount: len(PUNCT) + 3}
                            else:
                                # its not a special char create a word row
                                # most common case: if we have a keyword "because" occuring twice in a senctence
                                # this following row will add the word label to the second "because", which
                                # doesn't have a label till now
                                single_row = {Text: token, Label: WORD, Begin: begin + fulltext_offset,
                                              End: begin + len(token) + fulltext_offset,
                                              WordCount: 1,
                                              OffsetCount: len(WORD) + 3}
                            df = df.append(single_row, ignore_index = True)

    return df


def build_tree(df):
    """
        Takes a single sentence and builds the corresponding tree-structured string
        1. Start with the single word labels
        2. Add the word labels to the remaining words
        3. When looking at labels with wordcount >=3: call checkifRangeHasTwoChilds
            3.1 If true: continue
            3.2 If false: Depending on type of Label build the string with predefined rules
                3.2.1 SubCause, SubVariable, SubStatement, SubCondition --> rightbranching or leftbranching
                3.2.2 Cause-Effect-Relation with multiple Children
    """
    fulltext = df[df[Label].str.match(ROOT_SENTENCE)][Text].item()
    text = fulltext
    word_count_df = df
    fulltext_offset = df[df[Label].str.match(ROOT_SENTENCE)][Begin].item()
    root_sentence_token_count = df[df[Label].str.match(ROOT_SENTENCE)][WordCount].item()

    #Split . from the text
    if fulltext[-1] == ".":
        # add punct row
        punct_row = {Text: '.', Label: PUNCT, Begin: len(fulltext) - 1 + fulltext_offset,
                     End: len(fulltext) + fulltext_offset, WordCount: 1,
                     OffsetCount: len(PUNCT) + 3}
        word_count_df = word_count_df.append(punct_row, ignore_index = True)

        if fulltext[-2] == " ":
            # we have a space before the last punct
            if not root_has_two_childs(word_count_df, len(fulltext) - 1, fulltext_offset) and \
                    not root_has_two_childs(word_count_df, len(fulltext) - 2, fulltext_offset):
                # we have not a label from 0 to Punct and no label from 0 to Space
                # we want to have a label from 0 to space called sentence
                sentence_row = {Text: fulltext[:-2], Label: SENTENCE, Begin: 0 + fulltext_offset,
                                End: len(fulltext) - 2 + fulltext_offset,
                                WordCount: root_sentence_token_count - 1,
                                OffsetCount: len(SENTENCE) + 3}
                word_count_df = word_count_df.append(sentence_row, ignore_index = True)

        # if there exists a label which spans from 0 to PUNCT then we do not need to add another row
        # otherwise we need to add another row
        elif not root_has_two_childs(word_count_df, len(fulltext) - 1, fulltext_offset):
            sentence_row = {Text: fulltext[:-1], Label: SENTENCE, Begin: 0 + fulltext_offset,
                            End: len(fulltext) - 1 + fulltext_offset,
                            WordCount: root_sentence_token_count - 1,
                            OffsetCount: len(SENTENCE) + 3}
            word_count_df = word_count_df.append(sentence_row, ignore_index = True)

        # always increment the ROOT Sentencte with wordcount +=1
        word_count_df.loc[word_count_df[Label].str.match(ROOT_SENTENCE), [WordCount]] += 1
    elif fulltext[-1] == "!":
        # add ! row
        exclamation_row = {Text: '!', Label: PUNCT, Begin: len(fulltext) - 1 + fulltext_offset,
                           End: len(fulltext) + fulltext_offset, WordCount: 1,
                           OffsetCount: len(PUNCT) + 3}
        word_count_df = word_count_df.append(exclamation_row, ignore_index = True)

        # if there exists a label which spans from 0 to PUNCT then we do not need to add another row
        # otherwise we need to add another row
        if not root_has_two_childs(word_count_df, len(fulltext) - 1, fulltext_offset):
            sentence_row = {Text: fulltext[:-1], Label: SENTENCE, Begin: 0 + fulltext_offset,
                            End: len(fulltext) - 1 + fulltext_offset,
                            WordCount: root_sentence_token_count - 1,
                            OffsetCount: len(SENTENCE) + 3}
            word_count_df = word_count_df.append(sentence_row, ignore_index = True)

        # always increment the ROOT Sentencte with wordcount +=1
        word_count_df.loc[word_count_df[Label].str.match(ROOT_SENTENCE), [WordCount]] += 1

    word_count_df = word_count_df.sort_values(by = [WordCount], ascending = [True])
    indexes = word_count_df.index.values.tolist()

    single_labels = []
    single_labels_extended = []
    merged_token_indexes = []
    # Add entries to the dataframe
    # Such that we have - expect for the single word labels (Key_c) - a word label for each word
    for index in indexes:
        row = word_count_df.loc[index]
        if row[WordCount] == 1:
            # we have a label which we do not want to label with "WORD" add it to the array
            single_labels.append(row[Text])
            single_labels_extended.append((row[Text], row[Begin]))

    # tokenize with treebank and if one token is n't then merge it with the word before
    # tokenize input such that we know which parts of the fulltext are single words and to separate "(", ")", ","
    tokens = nltk.word_tokenize(text)
    print("TokenStart")
    print(tokens)
    print("TokenEnd")
    unused_indices = []
    counter = 0
    for index, token in enumerate(tokens):
        if token == "n't":
            tokens[index - 1] = tokens[index - 1] + "n't"
            merged_token_indexes.append(index - 1)
            unused_indices.append(index)
        elif token == "not":
            # check if the preceding token is can
            if tokens[index - 1] == "can":
                # count how many times we found this token pair
                counter += 1
                # check if we have cannot in the fulltext
                findings = [(i, fulltext[i:len("cannot") + i]) for i in findall("cannot", fulltext)]
                if len(findings) >= counter:
                    tokens[index - 1] = "cannot"
                    merged_token_indexes.append(index - 1)
                    unused_indices.append(index)
        elif token == "doesn":
            if tokens[index+1] == "’" and tokens[index+2] == "t":
                # we have a doesn't
                print("We have a doesn't in the sentence")
                tokens[index] = "doesn’t"
                merged_token_indexes.append(index)
                unused_indices.append(index+1)
                unused_indices.append(index + 2)
        elif token == "’" and (tokens[index + 1] == "s" or tokens[index + 1] == "S") and (
                not (tokens[index + 2] == "’")):
            if tokens[index + 1] == "S":
                tokens[index - 1] = tokens[index - 1] + "’S"
            else:
                tokens[index - 1] = tokens[index - 1] + "’s"
            unused_indices.append(index + 1)
            unused_indices.append(index)
            merged_token_indexes.append(index - 1)

        elif token == "'s" and (not tokens[index + 1] == "'"):
            tokens[index - 1] = tokens[index - 1] + "'s"
            unused_indices.append(index)
            merged_token_indexes.append(index - 1)
        elif token == "'S" and (not tokens[index + 1] == "'"):
            tokens[index - 1] = tokens[index - 1] + "'S"
            unused_indices.append(index)
            merged_token_indexes.append(index - 1)
        elif token == "e.g":
            tokens[index] = "e.g."
            unused_indices.append(index + 1)
        elif token == "i.e":
            tokens[index] = "i.e."
            unused_indices.append(index + 1)

    # before we remove the single labels add the single_token_rows where they match the single labels text
    # but they are not at the same position therefore include check for location shift
    word_count_df = check_if_in_single_labels(tokens, single_labels, single_labels_extended, fulltext, fulltext_offset,
                                              word_count_df)

    # after we have the remaining tokens calculate there position in the string and append a new row to the df
    for index_token, token in enumerate(tokens):
        if index_token not in unused_indices and token not in single_labels:
            if token == ",":
                word_count_df = create_single_token_row(fulltext, token, PUNCT, word_count_df, fulltext_offset, tokens,
                                                        merged_token_indexes)
            elif re.match(r'[^A-Za-z0-9]+', token) is not None:
                if token == "``":
                    word_count_df = create_single_token_row(fulltext, '"', SYMBOL, word_count_df, fulltext_offset,
                                                            tokens, merged_token_indexes)
                elif token == "''":
                    word_count_df = create_single_token_row(fulltext, '"', SYMBOL, word_count_df, fulltext_offset,
                                                            tokens, merged_token_indexes)
                else:
                    word_count_df = create_single_token_row(fulltext, token, SYMBOL, word_count_df, fulltext_offset,
                                                            tokens, merged_token_indexes)
            else:
                # Create Word row
                word_count_df = create_single_token_row(fulltext, token, WORD, word_count_df, fulltext_offset, tokens,
                                                        merged_token_indexes)


    # After call from createRow we have duplicate entries, remove them
    word_count_df.drop_duplicates(inplace = True)
    apostrophs = {'``', '"', '”', "''", "“", "”", "’"}

    temp_df = word_count_df
    mask = (temp_df[WordCount] > 2)
    temp_df = temp_df[mask]

    # these are the arrays for the insertion merges
    separator_merge_already_called = []

    indexes = temp_df.index.values.tolist()

    new_children = []

    for index in indexes:
        row = word_count_df.loc[index]
        children, parent_row = has_two_childs(index, word_count_df)
        if len(children) != 2:
            # No binary structure
            # Separator related
            for index2, child in enumerate(children):
                if child[3] == SYMBOL and child[0] == "-":
                    for i, x in enumerate(children[index2:]):
                        if i != 0 and (x[3] == SYMBOL and x[0] == "-"):
                            # two separators in one sentence
                            # merge the second separator with everything between the two separators and the result with the first separator

                            # check if we already merged the dash at child with one of the x in children[index2:]
                            if not index2 in separator_merge_already_called:
                                # call the separator_merge_function which merges everything in the array range defined by two indices

                                # the first index where we saw a seperator
                                first_index = index2
                                second_index = i + index2
                                if first_index == 0 and second_index == len(children) - 1:
                                    # the whole annotation is a seperated statement
                                    # None_Causal (e.g. due to varying magnetic field or current collection along the orbit)
                                    word_count_df, _ = separator_merge(first_index, second_index, children,
                                                                       word_count_df,
                                                                       fulltext,
                                                                       fulltext_offset, SYMBOL, entire_statement = True)
                                else:
                                    word_count_df, new_children = separator_merge(first_index, second_index, children,
                                                                                  word_count_df,
                                                                                  fulltext, fulltext_offset, SYMBOL)
                                separator_merge_already_called.append(first_index)
                elif child[3] == SYMBOL and child[0] == "(":
                    # there is a Parantheses, check if we have another parantheses in the whole children list, but not the first one
                    for i, x in enumerate(children[index2:]):
                        if i != 0 and (x[3] == SYMBOL and x[0] == ")"):
                            # two parantheses in one sentence
                            # merge the second separator with everything between the two separators and the result with the first separator

                            # check if we already merged the paranthese at child with one of the x in children[index2:]
                            if not index2 in separator_merge_already_called:
                                # call the separator_merge_function which merges everything in the array range defined by two indices

                                # the first index where we saw a seperator
                                first_index = index2
                                second_index = i + index2
                                if (first_index, second_index) not in separator_merge_already_called:
                                    if first_index == 0 and second_index == len(children) - 1:
                                        # the whole annotation is a seperated statement
                                        word_count_df, _ = separator_merge(first_index, second_index, children,
                                                                           word_count_df,
                                                                           fulltext,
                                                                           fulltext_offset, SYMBOL,
                                                                           entire_statement = True)
                                    else:
                                        word_count_df, new_children = separator_merge(first_index, second_index,
                                                                                      children,
                                                                                      word_count_df,
                                                                                      fulltext, fulltext_offset, SYMBOL)
                                    separator_merge_already_called.append(first_index)

                elif child[3] == SYMBOL and child[0] in apostrophs:
                    # there is a Parantheses, check if we have another parantheses in the whole children list, but not the first one
                    for i, x in enumerate(children[index2:]):
                        if i != 0 and (x[3] == SYMBOL and x[0] in apostrophs):
                            # two apostroph in one sentence
                            # merge the second apostroph with everything between the two separators and the result with the first apostroph

                            # check if we already merged the apostroph at child with one of the x in children[index2:]
                            if not index2 in separator_merge_already_called:
                                # call the separator_merge_function which merges everything in the array range defined by two indices

                                # the first index where we saw a seperator
                                first_index = index2
                                second_index = i + index2
                                if first_index == 0 and second_index == len(children) - 1:
                                    # the whole annotation is a seperated statement
                                    word_count_df, _ = separator_merge(first_index, second_index, children,
                                                                       word_count_df,
                                                                       fulltext,
                                                                       fulltext_offset, SYMBOL, entire_statement = True)
                                else:
                                    word_count_df, new_children = separator_merge(first_index, second_index, children,
                                                                                  word_count_df,
                                                                                  fulltext,
                                                                                  fulltext_offset, SYMBOL)
                                separator_merge_already_called.append(first_index)
                elif child[3] == SYMBOL and child[0] == "<":
                    for i, x in enumerate(children[index2:]):
                        if i != 0 and (x[3] == SYMBOL and x[0] == ">"):

                            # check if we already merged the gt  at child with one of the x in children[index2:]
                            if not index2 in separator_merge_already_called:
                                # the first index where we saw a seperator
                                first_index = index2
                                second_index = i + index2
                                if first_index == 0 and second_index == len(children) - 1:
                                    # the whole annotation is a seperated statement
                                    word_count_df, _ = separator_merge(first_index, second_index, children,
                                                                       word_count_df,
                                                                       fulltext,
                                                                       fulltext_offset, SYMBOL,
                                                                       entire_statement = True)
                                else:
                                    word_count_df, new_children = separator_merge(first_index, second_index, children,
                                                                                  word_count_df,
                                                                                  fulltext, fulltext_offset, SYMBOL)
                                separator_merge_already_called.append(first_index)
                elif child[3] == SYMBOL and child[0] == "[":
                    for i, x in enumerate(children[index2:]):
                        if i != 0 and (x[3] == SYMBOL and x[0] == "]"):
                            # check if we already merged the paranthese at child with one of the x in children[index2:]
                            if not index2 in separator_merge_already_called:
                                # the first index where we saw a seperator
                                first_index = index2
                                second_index = i + index2
                                if first_index == 0 and second_index == len(children) - 1:
                                    # the whole annotation is a seperated statement
                                    word_count_df, _ = separator_merge(first_index, second_index, children,
                                                                       word_count_df,
                                                                       fulltext,
                                                                       fulltext_offset, SYMBOL,
                                                                       entire_statement = True)
                                else:
                                    word_count_df, new_children = separator_merge(first_index, second_index, children,
                                                                                  word_count_df,
                                                                                  fulltext, fulltext_offset, SYMBOL)
                                separator_merge_already_called.append(first_index)
                elif child[3] == SYMBOL and child[0] == "{":
                    for i, x in enumerate(children[index2:]):
                        if i != 0 and (x[3] == SYMBOL and x[0] == "}"):
                            # check if we already merged the paranthese at child with one of the x in children[index2:]
                            if not index2 in separator_merge_already_called:
                                # the first index where we saw a seperator
                                first_index = index2
                                second_index = i + index2
                                if first_index == 0 and second_index == len(children) - 1:
                                    # the whole annotation is a seperated statement
                                    # None_Causal (e.g. due to varying magnetic field or current collection along the orbit)
                                    word_count_df, _ = separator_merge(first_index, second_index, children,
                                                                       word_count_df,
                                                                       fulltext,
                                                                       fulltext_offset, SYMBOL,
                                                                       entire_statement = True)
                                else:
                                    word_count_df, new_children = separator_merge(first_index, second_index, children,
                                                                                  word_count_df,
                                                                                  fulltext, fulltext_offset, SYMBOL)
                                separator_merge_already_called.append(first_index)

    temp_df2 = word_count_df
    mask = (temp_df2[WordCount] > 2)
    temp_df2 = temp_df2[mask]

    indexes3 = temp_df2.index.values.tolist()
    for index3 in indexes3:
        row = word_count_df.loc[index3]
        children, parent_row = has_two_childs(index3, word_count_df)
        if len(children) != 2:
            word_count_df = simple_merge(children, word_count_df, parent_row, fulltext, fulltext_offset)

    word_count_df = word_count_df.sort_values(by = [WordCount], ascending = [True])

    # Dataframe prepared, add every label to the fulltext
    indexes = word_count_df.index.values.tolist()

    for index in indexes:
        row = word_count_df.loc[index]
        # Insert Label at position, remove row and update Offsets on all other rows
        fulltext = fulltext[:(row[Begin] - fulltext_offset)] + "(" + row[Label] + " " + fulltext[
                                                                                        (row[Begin] - fulltext_offset):(
                                                                                                row[
                                                                                                    End] - fulltext_offset)] + ")" + fulltext[
                                                                                                                                     (
                                                                                                                                             row[
                                                                                                                                                 End] - fulltext_offset):]

        word_count_df.drop([index], inplace = True)
        # print(word_count_df.to_string())
        word_count_df = word_count_df.apply(update_offset, args = [row], axis = 1)
        print(word_count_df.to_string())

    return fulltext


def update_offset(updatedrow, row):
    """
        Takes a dataframe and updates all offsets according to the row specified by index
    """
    begin_offset = row[Begin]
    end_offset = row[End]
    offset = row[OffsetCount]

    if (updatedrow[Begin] <= begin_offset) and (updatedrow[End] >= end_offset):
        updatedrow[End] += offset

    if updatedrow[Begin] >= end_offset:
        updatedrow[Begin] += offset
        updatedrow[End] += offset

    return updatedrow


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_file', default = './data/Cause-Effect-Saetze.txt',
                        help = "Path to the file containing the annotations")
    return parser.parse_args()


if __name__ == '__main__':
    nltk.download('punkt')

    # move the current directory to the project root
    abspath = os.path.abspath(__file__)
    project_root_dir = os.path.join(os.path.dirname(abspath), '..')
    os.chdir(project_root_dir)

    # Parse the command line arguments
    args = parse_args()

    data = import_ann_file(args.txt_file)

    sorted_data = sort_annotations(data)

    # for each row tokenize the text entry and set the token size to the word count
    token_count = sorted_data[Text]
    column = token_count.apply(lambda x: len(nltk.word_tokenize(x)))
    print(column)
    sorted_data[WordCount] = column

    """
         - Add the offset count for each entry
         - We add + 3 here because we add "(", "\s", ")"
    """
    offset_count = sorted_data[Label].str.len() + 3
    sorted_data[OffsetCount] = offset_count

    print(sorted_data.head(7).to_string())
    sorted_data[WordCount] = pd.to_numeric(sorted_data[WordCount], errors = 'coerce')
    print(sorted_data.head(7).to_string())
    sorted_data[OffsetCount] = pd.to_numeric(sorted_data[OffsetCount], errors = 'coerce')
    print(sorted_data.head(7).to_string())

    file_name = args.txt_file[:-4]

    text_file = open(f"{file_name}-RNN.txt", "w")

    indexes = sorted_data.index.values.tolist()
    start_number = 0
    first = True
    for number, index in enumerate(indexes):
        row = sorted_data.loc[index]
        if row[Label] == ROOT_SENTENCE:
            if not first:
                sentence = sorted_data.iloc[start_number:number, :]

                text_file.write(build_tree(sentence) + '\n')

                start_number = number
            else:
                first = False
        if number == len(indexes) - 1:
            sentence = sorted_data.iloc[start_number:, :]
            text_file.write(build_tree(sentence))

    text_file.close()
