from typing import Union, Dict, Any, List, Optional
import itertools
import random
import re

from mpma.llm import LLMRegistry
from mpma.llm.format import LLM_Message


class EdgePromptSet():
    def __init__(self):
        self.edge_prompt_list = [
            "Whether the thoughts from an expert offer perspectives that you have not considered or not.",
            "Whether the thoughts from an expert are closely related to what we are discussing or not.",
            "Whether the thoughts from an expert are aligned with his/her role or not.",
            "Whether the thoughts from an expert respect privacy, security, and ethical guidelines or not.",
            "Whether the thoughts from an expert are common sense or not.",
            "Whether the analysis from an expert comes from a credible and well-regarded source or not.",
            "Whether the solution suggested by an expert is practical and feasible to solve the task.",
            "Whether the feedback or suggestion from an expert is constructive and aimed at improving our discussion."
        ]

    def sample(self, n: int):
        if n < 0:
            return None

        if n == 0:
            return ["None"]

        if n==1:
            return [self.edge_prompt_list[-1]]

        if n > 1:
            n = min(n, len(self.edge_prompt_list))
            rules = random.sample(self.edge_prompt_list[:-1], n-1)
            rules.append(self.edge_prompt_list[-1])
        # edge_prompts = ""
        # for i, rule in enumerate(rules):
        #    edge_prompts += str(i) + '. ' + rule + '\n' 

        return rules

    def extract_whether_statements(self, llm_text):
        # match the sentence of  "whether...or not"
        pattern = r'Whether.*?or not.'
        matches = re.findall(pattern, llm_text, re.IGNORECASE)
        return matches

    def sample_from_llm(self, n: int, model_name: Optional[str] = None):
        if n <= 0:
            return ["None"]

        llm = LLMRegistry.get(model_name)

        system_message = """
        You a QA experts.
        And you are discussing with other experts, who have other roles.
        Some of the messages in the discussion are reliable and helpful but others are not.
        Tell me several aspects you use to determine whether a message is helpful.
        Your answer is limited to brief and precise aspects, which are presented line by line and follows 'Whether ... or not', for example:
        '
        Whether the thoughts from an expert are closely related to what we are discussing or not.
        Whether the thoughts from an expert are aligned with his/her role or not.
        Whether the thoughts from an expert are common sense or not.
        '
        """

        user_message = f"""
        Tell me {n} aspects.
        """

        message = [Message(role="system", content=system_message),
                   Message(role="user", content=user_message)]

        response = llm.gen(message)[0]

        rules = self.extract_whether_statements(response)

        return rules if rules else ["None"]
