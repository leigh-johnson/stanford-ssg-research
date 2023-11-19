from enum import Enum
from abc import ABC, abstractmethod

from langchain.llms.base import BaseLLM
from langchain.pydantic_v1 import BaseModel
from langchain.schema.prompt_template import BasePromptTemplate

from llm_programs.models import InstructModel


class PromptTemplateType(str, Enum):
    DIRECT = "direct"
    COT = "cot"
    PROGRAM = "program"


def is_zero_shot_direct(llm: BaseLLM) -> bool:
    return llm.metadata.get("prompt_template_type", False) is PromptTemplateType.DIRECT


class BasePrompt(BaseModel, ABC):
    """
    This class is responsible for instantiating a BasePromptTemplate, based on number of examples and the requested prompt template type.
    """

    num_examples: int = 0
    prompt_template_type: PromptTemplateType
    instruct_model: InstructModel

    @abstractmethod
    def parse_final_answer(self) -> str:
        pass

    @abstractmethod
    def task_description(self) -> str:
        pass

    @abstractmethod
    def zero_shot_direct_prompt(self) -> BasePromptTemplate:
        pass

    @abstractmethod
    def zero_shot_cot_prompt(self, task_description="") -> BasePromptTemplate:
        pass

    @abstractmethod
    def few_shot_cot_prompt(self, num_examples: int, task_description="") -> BasePromptTemplate:
        pass

    @abstractmethod
    def zero_shot_program_prompt(self, task_description="") -> BasePromptTemplate:
        pass

    @abstractmethod
    def few_shot_program_prompt(self, num_examples: int, task_description="") -> BasePromptTemplate:
        pass

    def get_prompt(self, **kwargs) -> BasePromptTemplate:
        if self.prompt_template_type is PromptTemplateType.DIRECT:
            return self.zero_shot_direct_prompt()
        elif self.prompt_template_type is PromptTemplateType.COT:
            if self.num_examples == 0:
                return self.zero_shot_cot_prompt()
            else:
                return self.few_shot_cot_prompt(self.num_examples)
        elif self.prompt_template_type is PromptTemplateType.PROGRAM:
            if self.num_examples == 0:
                return self.zero_shot_program_prompt()
            else:
                return self.few_shot_program_prompt(self.num_examples)
        raise NotImplementedError(
            f"BasePromptSelector.get_prompt is not yet implemented for {self.prompt_template_type}"
        )
