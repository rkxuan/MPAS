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
from mpma.environment.agents.agent_registry import AgentRegistry


@OperationRegistry.register("CodeWriting")
class CodeWriting(Operation):
    def __init__(self, domain: str,
                 model_name: Optional[str] = None,
                 operation_description: str = "Address the given code writing problem.",
                 ):
        super().__init__(operation_description, None, True)
        self.domain = domain
        self.llm = LLMRegistry.get(model_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        
    @property
    def operation_name(self):
        return self.__class__.__name__

    async def _execute(self, raw_inputs: Message,
                       messages: List[Message],
                       self_messages: List[Message],
                       **kwargs):
        role_prompt = self.prompt_set.get_role_prompt(self.agent_role) 
        agent_role = self.prompt_set.get_role(self.agent_role)
        constraint = self.prompt_set.get_thought_constraint(self.agent_role)
        prompt = self.prompt_set.get_message_prompt(raw_inputs=raw_inputs, messages=messages, self_messages=self_messages, role=self.agent_role)
        llm_message = [LLM_Message(role="system", content=f"{role_prompt}{constraint}"),
                   LLM_Message(role="user", content=prompt)]
        response = await self.llm.agen(llm_message)
        return [Message(0, self.agent_id, agent_role, self.id, raw_inputs.task, response)]




