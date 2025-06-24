import pandas as pd
from typing import Dict, Any, Union, List
from collections.abc import Sequence
from abc import ABC, abstractmethod
from mpma.system import Message

class BaseDataset(ABC, Sequence[Any]):
    @staticmethod
    @abstractmethod
    def get_domain() -> str:
        """ To be overriden. """

    @abstractmethod
    def split(self) -> str:
        """ To be overriden. """

    @abstractmethod
    def record_to_system_input(self, record: pd.DataFrame) -> Message:
        """ To be overriden. """

    @abstractmethod
    def postprocess_answer(self, answer: List[Message]) -> str:
        """ To be overriden. """

    @abstractmethod
    def record_to_target_answer(self, record: pd.DataFrame) -> str:
        """ To be overriden. """
