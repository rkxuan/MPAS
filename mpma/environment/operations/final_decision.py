#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from collections import defaultdict, Counter
from enum import Enum
from typing import List, Any, Optional
import random

from mpma.llm.format import LLM_Message
from mpma.system import Operation, Message
from mpma.environment.prompt import PromptSetRegistry, PromptSet
from mpma.llm import LLMRegistry, LLM
from mpma.environment.operations.operation_registry import OperationRegistry


@OperationRegistry.register("FinalDecision")
class FinalDecision(Operation):
    def __init__(self, 
                 domain: str,
                 model_name: Optional[str],
                 operation_description: str = "Refer to all answers and give a final answer.",
                 ):
        super().__init__(operation_description, None, True)
        self.domain: str = domain
        self.llm: LLM = LLMRegistry.get(model_name)
        self.prompt_set: PromptSet = PromptSetRegistry.get(domain)

    @property
    def operation_name(self):
        return self.__class__.__name__
    
    async def _execute(self, raw_inputs: Message,
                       messages: List[Message], 
                       **kwargs) -> None:          
        #role_prompt = self.prompt_set.get_role("KnowledgeableExpert")
        role_prompt = self.prompt_set.get_role_prompt(role=self.agent_role)
        constraint = self.prompt_set.get_answer_constraint()
        prompt = self.prompt_set.get_message_prompt(raw_inputs=raw_inputs, messages=messages, self_messages=[Message(1)], role=self.agent_role)
        llm_message = [LLM_Message(role="system", content=f"{role_prompt}{constraint}"),
                   LLM_Message(role="user", content=prompt)]
        response = await self.llm.agen(llm_message)
        return [Message(0, None, None, self.id, raw_inputs.task, response)]
        
        
