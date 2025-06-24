#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import shortuuid
from typing import Tuple, Any, List, Optional, Dict
from copy import deepcopy
from abc import ABC, abstractmethod
import async_timeout
import numpy as np

from mpma.system.memory import Memory
from mpma.system.message import Message
from mpma.system.operation import Operation


class Agent(ABC):
    """
    An agent is composed of step-by-step operations based on LLMs. Each operation
    can perform specific operations, allowing for complex data processing workflows.
    The agent supports integration with language models, making it suitable for tasks
    that require natural language processing capabilities.

    Attributes:
        id (uuid.UUID): Unique identifier for the agent.
        domain (str): Unique identifier for the place which the agent is used for.
        globalmemory (Dict(Memory)): Dict maps task_id -> task_memory, and memory is the record of (raw_inputs, messages, outputs).
        model_name (str): The name of language model used for operations in the agent.
        operations (dict[str, Operation]): A collection of operations, each identified by a unique UUID.
        input_operations (list[uuid.UUID]]): List of operation ids designated as input points in the agent.
        output_operations (list[uuid.UUID]): List of operation ids designated as the primary output point in the agent.
        message_aggregation_operation (Operation): Operation to aggregate all the messages to the agent.

    Methods:
        build_agent():
            Method to be implemented for constructing the step-by-step operations of the agent.
        create_memory(key):
            Create a memory for the previous tasks, which is identified by the key.
        flush_memory():
            Clear all the memorys.
        add_operation(Operation):
            Add a new operation to the agent with a unique identifier.
        add_message_aggregation_operation(Operation):
            Add a message aggregation operation to the agent with a unique identifier.
        display(draw:bool):
            Displays a textual representation of the agent, with an option for a visual representation.
        chech_agent_structure():
            check whether something is wrong in the self-structure.
        message_aggregate(raw_inputs, messages, self_messages):
            aggregate the information of messages via MessageAggregate operation
        run(raw_inputs, messages, return_all_outputs):
            Executes the agent for a specified number of steps, processing provided inputs.
    """

    def __init__(self,
                 domain: str,
                 model_name: Optional[str] = None,
                 ):

        self.id = shortuuid.ShortUUID().random(length=4)
        self.domain = domain
        self.model_name = model_name
        self.globalmemory = {}
        self.operations = {}
        self.input_operations = {}
        self.output_operations = {}
        self.message_aggregation_operation = None  # t0 be added by the system
        self.build_agent()

    @property
    def num_edges(self):
        num_edges = 0
        for operation in self.operations.values():
            num_edges += len(operation.successors)
        return num_edges

    @property
    def num_operations(self):
        return len(self.operations)

    @property
    def agent_name(self):
        return self.__class__.__name__

    @abstractmethod
    def build_agent(self):
        """To be overriden by a descendant class"""
    
    def create_memory(self, key):
        for operation in self.operations.values():
            operation.globalmemory[key] = Memory()
        
        if self.message_aggregation_operation:
            self.message_aggregation_operation.globalmemory[key] = Memory()

        self.globalmemory[key] = Memory()

    def flush_memory(self):
        for operation in self.operations.values():
            operation.globalmemory = {}
        
        if self.message_aggregation_operation:
            self.message_aggregation_operation.globalmemory = {}

        self.globalmemory = {}

    def add_message_aggregation_operation(self, operation: Operation):
        assert operation.operation_name == "MessageAggregation", "The operation is not MessageAggregation, but added as MessageAggregation"
        operation_id = shortuuid.ShortUUID().random(length=4)
        while operation_id in self.operations:
            operation_id = shortuuid.ShortUUID().random(length=5)
        operation.id = operation_id

        operation.agent_id = self.id
        operation.agent_role = self.agent_name
        self.message_aggregation_operation = operation

    def add_operation(self, operation: Operation):
        operation_id = shortuuid.ShortUUID().random(length=4)
        while operation_id in self.operations:
            operation_id = shortuuid.ShortUUID().random(length=5)
        operation.id = operation_id

        operation.agent_id = self.id
        operation.agent_role = self.agent_name
        self.operations[operation_id] = operation

    def find_operation(self, id: str):
        for operation in self.operations.values():
            if operation.id == id:
                return operation
        raise Exception(f"Operation not found: {id} among "
                        f"{list(self.operations.keys())}")

    def check_agent_structure(self):
        def is_operation_useful(operation: Operation):
            if operation.id not in self.operations:
                return False

            for successor in operation.successors:
                if not is_operation_useful(successor):
                    return False
            return True

        for operation in self.operations.values():
            if not is_operation_useful(operation):
                print(f"{operation.operation_name} {operation.id} is not useful in the {self.agent_name} {self.id}")
                return False

        in_degrees = {operation_id: len(self.operations[operation_id].predecessors) for operation_id in self.operations}
        zero_in_degree_queue = [operation_id for operation_id, deg in in_degrees.items() if deg == 0]

        for input_operation in self.input_operations.values():
            if input_operation.id not in zero_in_degree_queue:
                print(
                    f"{input_operation.operation_name} {input_operation.id} as tht input operation in the {self.agent_name} {self.id}, but has predecessors")
                return False

        while zero_in_degree_queue:
            current_operation_id = zero_in_degree_queue.pop(0)
            current_operation = self.operations[current_operation_id]
            for successor in current_operation.successors:
                in_degrees[successor.id] -= 1
                if in_degrees[successor.id] == 0:
                    zero_in_degree_queue.append(successor.id)

        for operation_id, in_degree in in_degrees.items():
            if in_degree != 0:
                operation = self.operations[operation_id]
                print(
                    f"{operation.operation_name} {operation.id} will not be executed in the {self.agent_name} {self.id}")
                return False

        if not self.output_operations:
            print(f"without output operation in the {self.agent_name} {self.id}")
        return True


    async def message_aggregate(self, task_id: str, 
                                raw_inputs: Message,
                                messages: List[Message],
                                self_messages: List[Message],
                                edge_prompts: Optional[str] = None):
        await self.message_aggregation_operation.execute(task_id, raw_inputs, messages, self_messages, edge_prompts=edge_prompts)

    async def run(self, task_id: str, 
                  raw_inputs: Message,
                  messages: Optional[List[Message]] = None,
                  self_messages: Optional[List[Message]] = None,
                  return_all_outputs: bool = False,
                  **kwargs,
                  ):
        raw_inputs = deepcopy(raw_inputs)
        self.globalmemory[task_id].push_raw_inputs(raw_inputs)

        if messages:
            self.globalmemory[task_id].push_messages(messages)
        else:
            self.globalmemory[task_id].push_messages([Message(1)])
        if self_messages:
            self.globalmemory[task_id].push_self_messages(self_messages)
        else:
            self.globalmemory[task_id].push_self_messages([Message(1)])

        current_tasks = []
        current_operation_ids = list(self.input_operations.keys())
        in_degree = {operation_id: len(self.operations[operation_id].predecessors) for operation_id in self.operations}

        for input_operation in self.input_operations.values():
            current_tasks.append(asyncio.create_task(input_operation.execute(task_id, raw_inputs, messages, self_messages, **kwargs)))

        while current_tasks:
            await asyncio.gather(*current_tasks)
            current_tasks = []
            next_operation_ids = []

            for current_operation_id in current_operation_ids:
                current_operation = self.operations[current_operation_id]
                for successor in current_operation.successors:
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        current_tasks.append(asyncio.create_task(successor.execute(task_id, **kwargs)))
                        next_operation_ids.append(successor.id)
            current_operation_ids = next_operation_ids
        answers = []
        for output_operation in self.output_operations.values():
            output_messages = output_operation.globalmemory[task_id].outputs_buffer[-1]
            if not return_all_outputs:
                for output_message in output_messages:
                    if not output_message.EOF:
                        answers.append(output_message)
                        break
            else:
                for output_message in output_messages:
                    if not output_message.EOF:
                        answers.append(output_message)

        if not answers:
            answers = [Message(1)]
        self.globalmemory[task_id].push_outputs(answers)
        return self.globalmemory[task_id].outputs_buffer[-1]