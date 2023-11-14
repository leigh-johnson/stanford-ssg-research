from enum import Enum
from abc import ABC, abstractmethod

from langchain.llms.base import BaseLLM
from langchain.pydantic_v1 import BaseModel
from langchain.schema.prompt_template import BasePromptTemplate


class PromptTemplateType(str, Enum):
    ZERO_SHOT_DIRECT = "zero_shot_direct"
    FEW_SHOT_AUTO_COT = "few_shot_auto_cot"
    FEW_SHOT_TOOL = "few_shot_tool"


def is_zero_shot_direct(llm: BaseLLM) -> bool:
    return llm.metadata.get("prompt_template_type", False) is PromptTemplateType.ZERO_SHOT_DIRECT


class BasePrompt(BaseModel, ABC):
    """
    This class is responsible for instantiating a BasePromptTemplate, based on number of examples and the requested prompt template type.
    """

    num_examples: int = 0
    prompt_template_type: PromptTemplateType

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
    def few_shot_cot_prompt(self, task_description="") -> BasePromptTemplate:
        pass

    def get_prompt(self, **kwargs) -> BasePromptTemplate:
        if self.prompt_template_type is PromptTemplateType.ZERO_SHOT_DIRECT:
            return self.zero_shot_direct_prompt()
        elif self.prompt_template_type is PromptTemplateType.FEW_SHOT_AUTO_COT:
            if self.num_examples == 0:
                return self.zero_shot_cot_prompt()
            else:
                return self.few_shot_cot_prompt()
        raise NotImplementedError(
            f"BasePromptSelector.get_prompt is not yet implemented for {self.prompt_template_type}"
        )
