from abc import ABC, abstractmethod
import datasets

from langchain.pydantic_v1 import BaseModel
from langchain.schema.language_model import BaseLanguageModel
from llm_programs.prompts.base import PromptTemplateType, BasePrompt


class BaseTask(BaseModel, ABC):
    batch_size: int = 1
    dataset_revision: str = "main"
    dataset: str
    dataset_split: str = "test"
    instruct_model_id: str
    llm: BaseLanguageModel
    num_examples: int = 0
    prompt_template_type: PromptTemplateType
    prompt: BasePrompt
    streaming: bool = True
    verbose: bool = False

    def llmchain(self):
        prompt = self.prompt.get_prompt()
        return prompt | self.llm

    def load_dataset(self):
        return datasets.load_dataset(
            self.dataset,
            self.dataset_revision,
            streaming=self.streaming,
        )

    @abstractmethod
    def calc_accuracy(self, batch):
        pass

    @abstractmethod
    def calc_perplexity(self, batch):
        pass

    def calc_ptest(self):
        pass

    @abstractmethod
    def score(self, batch):
        pass

    def run(self):
        dataset = self.load_dataset()
        dataset = dataset[self.dataset_split]
        dataset = dataset.map(self.score)
        # .map(self.score_batch, batch_size=self.batch_size, batched=True)

        # llmchain = self.llmchain()
        # response = llmchain.batch(dataset, return_exceptions=True)
        for item in dataset:
            print(item)
            print("*****")
