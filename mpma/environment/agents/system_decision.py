from typing import Optional

from mpma.system import Agent
from mpma.environment.operations import FinalDecision
from mpma.environment.agents.agent_registry import AgentRegistry
from mpma.environment.prompt import PromptSetRegistry


@AgentRegistry.register('SystemDecision')
class SystemDecision(Agent):             # as the output agent in the system
    def __init__(self, domain: str,
                 model_name: Optional[str] = None):
        super().__init__(domain, model_name)

    def build_agent(self):
        final_decision_operation = FinalDecision(self.domain, self.model_name)
        self.add_operation(final_decision_operation)
        self.input_operations[final_decision_operation.id] = final_decision_operation
        self.output_operations[final_decision_operation.id] = final_decision_operation
