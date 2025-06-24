from typing import List, Optional
import shortuuid

from mpma.system.message import Message


class Memory():                  # Memory records the behaviour of operations and agents.
    def __init__(self):
        self.raw_inputs: Optional[Message] = None             # Description of the user question (Message)
        self.self_messages_buffer: List[List[Message]] = []   # Messages from self (List[Message])
        self.messages_buffer: List[List[Message]]  = []       # Messages from neighbor operations (List[Message])
        self.outputs_buffer: List[List[Message]] = []         # Operation outputs (List[Message])
    
    def push_raw_inputs(self, raw_inputs: Message):
        self.raw_inputs = raw_inputs

    def push_self_messages(self, self_messages: List[Message]):
        self.self_messages_buffer.append(self_messages)

    def push_messages(self, messages: List[Message]):
        self.messages_buffer.append(messages) 

    def push_outputs(self, outputs: List[Message]):
        self.outputs_buffer.append(outputs)

    def printmessages(self, index):        
        messages = self.messages_buffer[index]
        self_messages = self.self_messages_buffer[index]
        outputs = self.outputs_buffer[index]

        text = ""
        if self_messages[0].EOF:
            self_text = "(1) Self_message: None\n"
        else:
            self_text = "(1) Self_message:\n" + self_messages[0].message2print()

        if messages[0].EOF:
            message_text = "(2) Messages: None\n"
        else:
            message_text = "(2) Messages:\n"
            for i, message in enumerate(messages):
                message_text = message_text + message.message2text(i)

        if outputs[0].EOF:
            output_text = "(3) Outputs: None"
        else:
            output_text = "(3) Outputs: \n"
            for i, output in enumerate(outputs):
                output_text = output_text + output.message2text(i)
        
        return text + self_text + message_text + output_text

    def flush(self):
        self.raw_inputs = None
        self.messages = []
        self.outputs = []
