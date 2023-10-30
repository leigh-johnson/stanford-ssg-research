from abc import ABC, abstractmethod
import datasets

from langchain.pydantic_v1 import BaseModel
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.prompt_template import BasePromptTemplate
from llm_programs.prompts.base import PromptTemplateType
from langchain.chains.prompt_selector import ConditionalPromptSelector


class BaseTask(BaseModel, ABC):
    dataset: str
    dataset_revision: str = "main"
    llm: BaseLanguageModel
    num_examples: int = 0
    prompt_template_type: PromptTemplateType
    prompt_selector: ConditionalPromptSelector
    streaming: bool = True
    verbose: bool = False

    @abstractmethod
    def task_description(self) -> str:
        pass

    def llmchain(self):
        prompt = self.prompt_selector.get_prompt(self.llm)
        return prompt | self.llm

    def load_dataset(self):
        return datasets.load_dataset(
            self.dataset,
            self.dataset_revision,
            streaming=self.streaming,
        )

    def run(self):
        dataset = self.load_dataset()
