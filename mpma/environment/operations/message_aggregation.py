#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from collections import defaultdict, Counter
from typing import List, Any, Optional
import re

from mpma.llm.format import LLM_Message
from mpma.system import Operation, Message
from mpma.environment.prompt import PromptSetRegistry, PromptSet
from mpma.environment.prompt.common import get_message_prompt
from mpma.llm import LLMRegistry, LLM
from mpma.environment.operations.operation_registry import OperationRegistry


@OperationRegistry.register("MessageAggregation")
class MessageAggregation(Operation):
    def __init__(self,
                 domain: str,
                 model_name: Optional[str],
                 aggregation_strategy: str,
                 operation_description: str = "Refer to received messages from neighbors, aggregate/pool useful information from messages.",
                 ):
        super().__init__(operation_description, None, True)
        self.domain = domain
        self.llm = LLMRegistry.get(model_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.aggregation_strategy = aggregation_strategy

    @property
    def operation_name(self):
        return self.__class__.__name__

    async def _execute(self, raw_inputs: Message,
                       messages: List[Message],
                       self_messages: List[Message],
                       edge_prompts: Optional[str] = None,
                       **kwargs) -> list[Any]:
        if messages[0].EOF or self.agent_role == 'Critic' or self.aggregation_strategy == "Concat":
            return messages

        elif self.aggregation_strategy == "EdgeSelection":
            role_prompt = self.prompt_set.get_role_prompt(self.agent_role)
            constraint = self.prompt_set.get_edge_selection_constraint(edge_prompts)
            prompt = self.prompt_set.get_message_prompt(raw_inputs=raw_inputs, messages=messages,
                                                        self_messages=self_messages)
            llm_message = [LLM_Message(role="system", content=f"{role_prompt}{constraint}"),
                           LLM_Message(role="user", content=prompt)]
            response = await self.llm.agen(llm_message)
            response = response.strip('[]')
            numbers = response.split(',')
            aggregated_messages = []
            for number in numbers:
                try:
                    num = int(number.strip())
                    if 0 < num <= len(messages):
                        aggregated_messages.append(messages[num - 1])
                except ValueError:
                    continue
            return aggregated_messages if aggregated_messages else [Message(1)]
        
        elif self.aggregation_strategy == 'AttentionPooling':
            role_prompt = self.prompt_set.get_role_prompt(self.agent_role)
            constraint = self.prompt_set.get_attention_pooling_constraint()
            prompt = self.prompt_set.get_message_prompt(raw_inputs=raw_inputs, messages=messages,
                                                        self_messages=self_messages)
            llm_message = [LLM_Message(role="system", content=f"{role_prompt}{constraint}"),
                           LLM_Message(role="user", content=prompt)]
            response = await self.llm.agen(llm_message)
            aggregated_messages = []
            pattern = r"message id: (\d+); (.*)"
            #print("check respond: ", response,"\n")
            for line in response.splitlines():
                match = re.search(pattern, line)
                if match:
                    message_id = int(match.group(1))
                    information = match.group(2)
                    #print("check re: ", message_id, " ", information, "\n")
                    if 0 < message_id <= len(messages):
                        aggregated_message = messages[message_id - 1]
                        aggregated_message.textual_output = information
                        aggregated_messages.append(aggregated_message)
            return aggregated_messages if aggregated_messages else [Message(1)]

        elif self.aggregation_strategy == 'EdgeAttentionPooling':
            role_prompt = self.prompt_set.get_role_prompt(self.agent_role)
            constraint = self.prompt_set.get_edge_attention_pooling_constraint(edge_prompts)
            prompt = self.prompt_set.get_message_prompt(raw_inputs=raw_inputs, messages=messages,
                                                        self_messages=self_messages)
            llm_message = [LLM_Message(role="system", content=f"{role_prompt}{constraint}"),
                           LLM_Message(role="user", content=prompt)]
            response = await self.llm.agen(llm_message)
            aggregated_messages = []
            pattern = r"message id: (\d+); (.*)"
            #print("check respond: ", response,"\n")
            for line in response.splitlines():
                match = re.search(pattern, line)
                if match:
                    message_id = int(match.group(1))
                    information = match.group(2)
                    #print("check re: ", message_id, " ", information, "\n")
                    if 0 < message_id <= len(messages):
                        aggregated_message = messages[message_id - 1]
                        aggregated_message.textual_output = information
                        aggregated_messages.append(aggregated_message)
            return aggregated_messages if aggregated_messages else [Message(1)]

        else:
            raise ValueError(f"No such aggregation_strategy {self.aggregation_strategy}")
        
        """
        elif self.aggregation_strategy == 'AttentionPooling':
            role_prompt = self.prompt_set.get_role_prompt(self.agent_role)
            constraint = self.prompt_set.get_attention_pooling_constraint()
            prompt = self.prompt_set.get_message_prompt(raw_inputs=raw_inputs, messages=messages,
                                                        self_messages=self_messages)
            llm_message = [LLM_Message(role="system", content=f"{role_prompt}{constraint}"),
                           LLM_Message(role="user", content=prompt)]
            response = await self.llm.agen(llm_message)
            aggregated_messages = []
            pattern = r'message id: (\d+), expert id: (.*), information: (.*)'
            #print("check respond: ", response,"\n")
            for line in response.splitlines():
                match = re.search(pattern, line)
                if match:
                    message_id = int(match.group(1))
                    agent_id = match.group(2)
                    information = match.group(3).strip()
                    #print("check re: ", message_id, " ", agent_id, " ", information, "\n")
                    if 0 < message_id <= len(messages):
                        aggregated_message = messages[message_id - 1]
                        if aggregated_message.agent_id == agent_id:
                            aggregated_message.textual_output = information
                            aggregated_messages.append(aggregated_message)
            return aggregated_messages if aggregated_messages else [Message(1)]
            
        elif self.aggregation_strategy == 'EdgeAttentionPooling':
            role_prompt = self.prompt_set.get_role_prompt(self.agent_role)
            constraint = self.prompt_set.get_edge_attention_pooling_constraint(edge_prompts)
            prompt = self.prompt_set.get_message_prompt(raw_inputs=raw_inputs, messages=messages,
                                                        self_messages=self_messages)
            llm_message = [LLM_Message(role="system", content=f"{role_prompt}{constraint}"),
                           LLM_Message(role="user", content=prompt)]
            response = await self.llm.agen(llm_message)
            aggregated_messages = []
            pattern = r'message id: (\d+), expert id: (.*), information: (.*)'
            if response:
                for line in response.splitlines():
                    match = re.search(pattern, line)
                    if match:
                        message_id = int(match.group(1))
                        agent_id = match.group(2)
                        information = match.group(3).strip()
                        if 0 < message_id <= len(messages):
                            aggregated_message = messages[message_id - 1]
                            if aggregated_message.agent_id == agent_id:
                                aggregated_message.textual_output = information
                                aggregated_messages.append(aggregated_message)
            return aggregated_messages if aggregated_messages else [Message(1)]
        """
        
