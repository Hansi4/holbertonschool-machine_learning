#!/usr/bin/env python3
""" Question Answering """


import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """ A python function that finds a snippet of
    text within a reference document to answer a question """

    # Specialized for the SQuAD (Stanford Question Answering Dataset) task
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')

    # Predict the start and end positions of an answer in a text passage
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # Breaking the text down into smaller units (tokens) that the model can understand
    quest_tokens = tokenizer.tokenize(question)
    refer_tokens = tokenizer.tokenize(reference)

    # Preparation of Input Sequence. Add special tokens to include "classification" and "separator"
    tokens = ['[CLS]'] + quest_tokens + ['[SEP]'] + refer_tokens + ['[SEP]']

    # The tokens are converted into numerical IDs
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)

    # A list of 1 indicates the presence of a token
    # Used to differentiate between tokens and padding
    input_mask = [1] * len(input_word_ids)

    # 0 for question segment, 1 for reference segments
    input_type_ids = [0] * (1 + len(quest_tokens) + 1) + [1] * (len(refer_tokens) + 1)

    # Convert the input data to TF tensors, with additional batch
    # Used to provide the data to BERT model
    input_word_ids, input_mask, input_type_ids = map(
        lambda t: tf.expand_dims(
            tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (input_word_ids, input_mask, input_type_ids))

    # call the bert model
    outputs = model([input_word_ids, input_mask, input_type_ids])

    # find the positions of the start and end 
    # of predicted answer span in model outputs
    # output[0] represents the logits (predictions) for the start position of the answer
    # output[0][0] indexes into the first (and only) batch
    # output[0][0][1] identifies the index of the maximum value in the sliced tensor.
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    if answer == None or answer == "" or question in answer:
        return None

    return answer
