from typing import Optional

from mpma.system import Agent
from mpma.environment.operations import MathAnalyze
from mpma.environment.agents.agent_registry import AgentRegistry
from mpma.environment.prompt import PromptSetRegistry

@AgentRegistry.register('MathSolver')
class MathSolver(Agent):
    def __init__(self, domain: str,
                 model_name: Optional[str] = None):
        super().__init__(domain, model_name)

    def build_agent(self):
        qa_operation = MathAnalyze(self.domain, self.model_name)
        self.add_operation(qa_operation)
        self.input_operations[qa_operation.id] = qa_operation
        self.output_operations[qa_operation.id] = qa_operation