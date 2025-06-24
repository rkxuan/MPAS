import re
from typing import List, Union, Literal, Iterator, Any

from mpma.system import Message

def record_to_system_input(record):
    return Message(False, None, None, None, record["task"], None)

def humaneval_data_process(dataset):
    list_data_dict = []
    for data in dataset:
        task = data["prompt"]
        test = data["test"]
        list_data_dict.append({"task": task, "test": test})
    return list_data_dict

def humaneval_answer_process(answers:List[Message]):
    final_answer = ""
    for answer in answers:
        if not answer.EOF:
            final_answer = answer.textual_output

    final_answer = final_answer.lstrip("```python\n").rstrip("\n```")
    return final_answer