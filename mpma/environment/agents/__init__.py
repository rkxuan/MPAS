from mpma.environment.agents.mmlu.adversarial import Adversarial
from mpma.environment.agents.mmlu.critic import Critic
from mpma.environment.agents.mmlu.doctor import Doctor
from mpma.environment.agents.mmlu.economist import Economist
from mpma.environment.agents.mmlu.historian import Historian
from mpma.environment.agents.mmlu.lawyer import Lawyer
from mpma.environment.agents.mmlu.mathematician import Mathematician
from mpma.environment.agents.mmlu.programmer import Programmer
from mpma.environment.agents.mmlu.psychologist import Psychologist

from mpma.environment.agents.gsm8k.inspector  import Inspector
from mpma.environment.agents.gsm8k.math_programming_expert import MathProgrammingExpert
from mpma.environment.agents.gsm8k.math_solver import MathSolver
from mpma.environment.agents.gsm8k.mathematical_analyst import MathematicalAnalyst

from mpma.environment.agents.humaneval.algorithm_designer import AlgorithmDesigner
from mpma.environment.agents.humaneval.bug_fixer import BugFixer
from mpma.environment.agents.humaneval.programming_expert import ProgrammingExpert
from mpma.environment.agents.humaneval.project_manager import ProjectManager
from mpma.environment.agents.humaneval.test_analyst import TestAnalyst

from mpma.environment.agents.system_decision import SystemDecision
from mpma.environment.agents.knowledgeable_expert import KnowledgeableExpert
from mpma.environment.agents.agent_registry import AgentRegistry

__all__ = [
    "Adversarial",
    "Critic",
    "Doctor",
    "Economist",
    "Historian",
    "Lawyer",
    "Mathematician",
    "Programmer",
    "Psychologist",

    "Inspector",
    "MathProgrammingExpert",
    "MathSolver",
    "MathematicalAnalyst",

    "AlgorithmDesigner",
    "BugFixer",
    "ProgrammingExpert",
    "ProjectManager",
    "TestAnalyst",

    "SystemDecision",
    "AgentRegistry",
    "KnowledgeableExpert",
]