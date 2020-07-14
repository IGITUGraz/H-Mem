"""Utilities for downloading and parsing bAbI task data.

Modified from https://github.com/domluna/memn2n/blob/master/data_utils.py.

"""

import os
import re
import shutil
import urllib.request

import numpy as np

tasks = {
    1: 'single_supporting_fact',
    2: 'two_supporting_facts',
    3: 'three_supporting_facts',
    4: 'two_arg_relations',
    5: 'three_arg_relations',
    6: 'yes_no_questions',
    7: 'counting',
    8: 'lists_sets',
    9: 'simple_negation',
    10: 'indefinite_knowledge',
    11: 'basic_coreference',
    12: 'conjunction',
    13: 'compound_coreference',
    14: 'time_reasoning',
    15: 'basic_deduction',
    16: 'basic_induction',
    17: 'positional_reasoning',
    18: 'size_reasoning',
    19: 'path_finding',
    20: 'agents_motivations'
}


def download(extract=True):
    """Downloads the data set.

    Arguments:
      extract: boolean, whether to extract the downloaded archive (default=`True`).

    Returns:
      data_dir: string, the data directory.

    """
    url = 'https://s3.amazonaws.com/text-datasets/'
    file_name = 'babi_tasks_1-20_v1-2.tar.gz'
    data_dir = 'data/'
    file_path = data_dir + file_name

    if not os.path.exists(file_path):
        print('Downloading ' + url + file_name + '...')
        print('-')
        with urllib.request.urlopen(url + file_name) as response, open(file_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        shutil.unpack_archive(file_path, data_dir)
        shutil.move(data_dir + 'tasks_1-20_v1-2', data_dir + 'babi_tasks_1-20_v1-2')

    return data_dir + 'babi_tasks_1-20_v1-2'


def load_task(data_dir, task_id, training_set_size='1k', only_supporting=False):
    """Loads the nth task. There are 20 tasks in total.

    Arguments:
      data_dir: string, the data directory.
      task_id: int, the ID of the task (valid values are in `range(1, 21)`).
      training_set_size: string, the size of the training set to load (`1k` or `10k`, default=`1k`).
      only_supporting: boolean, if `True` only supporting facts are loaded (default=`False`).

    Returns:
      A Python tuple containing the training and testing data for the task.

    """
    assert task_id > 0 and task_id < 21

    data_dir = data_dir + '/en/' if training_set_size == '1k' else data_dir + '/en-10k/'
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'qa{}_'.format(task_id)
    train_file = [f for f in files if s in f and 'train' in f][0]
    test_file = [f for f in files if s in f and 'test' in f][0]
    train_data = _get_stories(train_file, only_supporting)
    test_data = _get_stories(test_file, only_supporting)

    return train_data, test_data


def vectorize_data(data, word_idx, max_num_sentences, sentence_size, query_size):
    """Vectorize stories, queries and answers.

    If a sentence length < `sentence_size`, the sentence will be padded with `0`s. If a story length <
    `max_num_sentences`, the story will be padded with empty sentences. Empty sentences are 1-D arrays of
    length `sentence_size` filled with `0`s. The answer array is returned as a one-hot encoding.

    Arguments:
      data: iterable, containing stories, queries and answers.
      word_idx: dict, mapping words to unique integers.
      max_num_sentences: int, the maximum number of sentences to extract.
      sentence_size: int, the maximum number of words in a sentence.
      query_size: int, the maximum number of words in a query.

    Returns:
      A Python tuple containing vectorized stories, queries, and answers.

    """
    S = []
    Q = []
    A = []
    for story, query, answer in data:
        if len(story) > max_num_sentences:
            continue

        ss = []
        for i, sentence in enumerate(story, 1):
            # Pad to sentence_size, i.e., add nil words, and add story.
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)

        # Make the last word of each sentence the time 'word' which corresponds to vector of lookup table.
        for i in range(len(ss)):
            ss[i][-1] = len(word_idx) - max_num_sentences - i + len(ss)

        # Pad stories to max_num_sentences (i.e., add empty stories).
        ls = max(0, max_num_sentences - len(ss))
        for _ in range(ls):
            ss.append([0] * sentence_size)

        # Pad queries to query_size (i.e., add nil words).
        lq = max(0, query_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        y = np.zeros(len(word_idx) + 1)  # 0 is reserved for nil word.
        for a in answer:
            y[word_idx[a]] = 1

        S.append(ss)
        Q.append(q)
        A.append(y)

    return np.array(S), np.array(Q), np.array(A)


def _get_stories(f, only_supporting=False):
    """Given a file name, read the file, retrieve the stories, and then convert the sentences into a single
    story.

    If only_supporting is true, only the sentences that support the answer are kept.

    Arguments:
      f: string, the file name.
      only_supporting: boolean, if `True` only supporting facts are loaded (default=`False`).

    Returns:
      A list of Python tuples containing stories, queries, and answers.

    """
    with open(f) as f:
        data = _parse_stories(f.readlines(), only_supporting=only_supporting)

        return data


def _parse_stories(lines, only_supporting=False):
    """Parse stories provided in the bAbI tasks format.

    If only_supporting is true, only the sentences that support the answer are kept.

    Arguments:
      lines: iterable, containing the sentences of a full story (story, query, and answer).
      only_supporting: boolean, if `True` only supporting facts are loaded (default=`False`).

    Returns:
      A Python list containing the parsed stories.

    """
    data = []
    story = []
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:  # Question
            q, a, supporting = line.split('\t')
            q = _tokenize(q)
            a = [a]  # Answer is one vocab word even ie it's actually multiple words.
            substory = None

            # Remove question marks
            if q[-1] == '?':
                q = q[:-1]

            if only_supporting:
                # Only select the related substory.
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories.
                substory = [x for x in story if x]

            data.append((substory, q, a))
            story.append('')
        else:  # Regular sentence
            sent = _tokenize(line)
            # Remove periods
            if sent[-1] == '.':
                sent = sent[:-1]
            story.append(sent)

    return data


def _tokenize(sent):
    """Return the tokens of a sentence including punctuation.

    Arguments:
      sent: iterable, containing the sentence.

    Returns:
      A Python list containing the tokens in the sentence.

    Examples:

    ```python
    tokenize('Bob dropped the apple. Where is the apple?')

    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    ```

    """
    return [x.strip() for x in re.split(r'(\W+)+?', sent) if x.strip()]
