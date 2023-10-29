from enum import Enum
from abc import ABC, abstractmethod
import datasets

from langchain.pydantic_v1 import BaseModel
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.prompt_template import BasePromptTemplate
from langchain.chains.llm import LLMChain
from llm_programs.prompts import PromptTemplateType


class BaseTask(BaseModel, ABC):
    dataset: str
    dataset_revision: str = "main"
    llm: BaseLanguageModel
    num_examples: int = 0
    prompt_template_type: PromptTemplateType
    streaming: bool = True
    verbose: bool = False

    @abstractmethod
    def prompt_template_cls(self) -> BasePromptTemplate:
        pass

    def llmchain(self):
        prompt = self.prompt_template_cls()
        return LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)

    def load_dataset(self):
        return datasets.load_dataset(
            self.dataset,
            self.dataset_revision,
            streaming=self.streaming,
        )

    def run(self):
        dataset = self.load_dataset()
