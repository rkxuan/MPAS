from typing import List, Optional, Dict
from mpma.system import Message


def get_message_prompt(raw_inputs: Message, messages: List[Message], self_messages: List[Message]):
    question_prompt = "Question: " + raw_inputs.task + "\n"

    if self_messages[0].EOF:
        self_prompt = "Your thoughts: None\n"
    else:
        self_prompt = "Your thoughts:\n" + self_messages[0].message2text(0)

    if messages[0].EOF:
        message_prompt = "Messages: None"
    else:
        message_prompt = "Message:\n"
        for i, message in enumerate(messages):
            message_prompt = message_prompt + message.message2text(i+1)
    return question_prompt + self_prompt + message_prompt


