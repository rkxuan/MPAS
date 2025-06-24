#!/usr/bin/env python
# -*- coding: utf-8 -*-

import shortuuid
import asyncio
from typing import List, Any, Optional
from abc import ABC, abstractmethod
import warnings
from copy import deepcopy

from mpma.system.message import Message
from mpma.system.memory import Memory
import pdb


class Operation(ABC):
    """
    Represents a processing unit within an agent.

    This class encapsulates the functionality for an operation in an agent, managing
    connections to other operations, handling inputs and outputs, and executing
    assigned operations asynchronously. It supports both individual and
    aggregated processing modes.

    Attributes:
        id (uuid.UUID): Unique identifier for the operation.
        globalmemory (Dict(Memory)): Dict maps task_id -> task_memory, and memory is the record of (raw_inputs, messages, outputs).
        operation_description (str): Brief description of this operation.
        predecessors (List[Operation]): Operations that precede this operation in the agent.
        successors (List[Operation]): Operations that succeed this operation in the agent.
        combine_messages_as_one (bool): The tag to decide whether process messages one by one.
        agent_id (uuid.UUID): Unique Identifier for the agent to which the operation belongs
        agent_role (str): Description for the role of the agent

    Methods:
        add_predecessor (Operation):
            Adds an operation as a predecessor of this operation, establishing a directed connection.
        add_successor (Operation):
            Adds an operation as a successor of this operation, establishing a directed connection.
        remove_predecessor (Operation):
            removes an operation as a predecessor of this operation.
        remove_successor (Operation):
            removes an operation as a successor of this operation.
        process_inputs(raw_inputs, messages, self_messages):
            pre-processes the input data.
        execute (**kwargs):
            Asynchronously processes the inputs through the operation, handling each input individually.
        _execute (input, **kwargs):
            An internal method that defines how a single input is processed by the operation.
            This method should be implemented specifically for each operation type.
    """

    def __init__(self,
                 operation_description: str,
                 id: Optional[str] = None,
                 combine_messages_as_one: bool = True):
        """
        Initializes a new operation instance in an agent.
        """
        self.id = id
        self.globalmemory = {}
        self.operation_description = operation_description
        self.predecessors: List[Operation] = []
        self.successors: List[Operation] = []
        self.combine_messages_as_one = combine_messages_as_one
        self.agent_id = None
        self.agent_role = None

    @property
    def operation_name(self):
        return self.__class__.__name__

    def add_predecessor(self, operation: 'Operation'):

        if operation not in self.predecessors:
            self.predecessors.append(operation)
            operation.successors.append(self)

    def add_successor(self, operation: 'Operation'):

        if operation not in self.successors:
            self.successors.append(operation)
            operation.predecessors.append(self)

    def remove_predecessor(self, operation: 'Operation'):
        if operation in self.predecessors:
            self.predecessors.remove(operation)
            operation.successors.remove(self)

    def remove_successor(self, operation: 'Operation'):
        if operation in self.successors:
            self.successors.remove(operation)
            operation.predecessors.remove(self)

    def process_inputs(self, task_id: str, 
                       raw_inputs: Optional[Message] = None,
                       messages: Optional[List[Message]] = None,
                       self_messages: Optional[List[Message]] = None):
        
        if not messages:
            messages = []
            for predecessor in self.predecessors:
                messages.extend(predecessor.globalmemory[task_id].outputs_buffer[-1])

        if not raw_inputs:
            if self.globalmemory[task_id].raw_inputs:
                raw_inputs = self.globalmemory[task_id].raw_inputs
            elif self.predecessors:
                raw_inputs = self.predecessors[0].globalmemory[task_id].raw_inputs
        
        if raw_inputs:
            raw_inputs = deepcopy(raw_inputs)
        else:
            raise ValueError(f"No any description of the task to operation {self.id} in agent {self.agent_id}")

        if messages:
            messages = deepcopy(messages)
            useful_messages = []
            for message in messages:
                if not message.EOF:
                    useful_messages.append(message)
            messages = useful_messages if useful_messages else [Message(1)]
        else:
            messages = [Message(1)]
        
        if self_messages:
            self_messages = deepcopy(self_messages)
            useful_self_messages = []
            for message in self_messages:
                if not message.EOF:
                    useful_self_messages.append(message)
                    break
            self_messages = useful_self_messages if useful_self_messages else [Message(1)]
        else:
            self_messages = [Message(1)]

        self.globalmemory[task_id].push_raw_inputs(raw_inputs)
        self.globalmemory[task_id].push_messages(messages)
        self.globalmemory[task_id].push_self_messages(self_messages)

        return raw_inputs, messages, self_messages

    async def execute(self, task_id: str, 
                      raw_inputs: Message,
                      messages: Optional[List[Message]] = None,
                      self_messages: Optional[List[Message]] = None,
                      **kwargs):
        tasks = []

        raw_inputs, messages, self_messages = self.process_inputs(task_id, raw_inputs, messages, self_messages)

        if self.combine_messages_as_one:
            tasks.append(asyncio.create_task(
                self._execute(raw_inputs=raw_inputs, messages=messages, self_messages=self_messages, **kwargs)))
        else:
            for message in messages:
                tasks.append(asyncio.create_task(
                    self._execute(raw_inputs=raw_inputs, messages=[message], self_messages=self_messages, **kwargs)))
        outputs = []
        if tasks:
            output_messages_list = await asyncio.gather(*tasks)
            for output_messages in output_messages_list:
                for output_message in output_messages:
                    if not output_message.EOF:
                        outputs.append(output_message)
        if not outputs:
            outputs = [Message(1)]
        self.globalmemory[task_id].push_outputs(outputs)
        
    @abstractmethod
    async def _execute(self, input, **kwargs):
        """ To be overriden by the descendant class """

