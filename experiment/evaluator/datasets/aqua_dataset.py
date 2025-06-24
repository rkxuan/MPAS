import re
from typing import List, Union, Literal, Iterator, Any
import random

from mpma.system import Message

def record_to_system_input(record):
    return Message(False, None, None, None, record["task"], None)

def aqua_data_process(dataset):
    # extract the question, step and answer
    list_data_dict = []
    for data in dataset:
        question = data['question']
        for option in data['options']:
            question += "\n" + option
        item = {"task": question}
        item["answer"] = data['correct']
        list_data_dict.append(item)

    return list_data_dict

def aqua_get_predict(answers:List[Message]):
    pred_str = None
    for answer in answers:
        if not answer.EOF:
            pred_str = answer.textual_output

    if not pred_str:
        raise Exception("Without any useful textual output")

    if('The answer is ' in pred_str):
        pred = pred_str.split('The answer is ')[-1].strip()
    elif('the answer is ' in pred_str):
        pred = pred_str.split('the answer is ')[-1].strip()
    else:
        pred = random.choice(['A', 'B', 'C', 'D', 'E'])

    return pred[0]

