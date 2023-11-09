from abc import ABC, abstractmethod
import datasets

from langchain.pydantic_v1 import BaseModel
from langchain.schema.language_model import BaseLanguageModel
from llm_programs.prompts.base import PromptTemplateType, BasePrompt


class BaseTask(BaseModel, ABC):
    batch_size: int = 1
    dataset_revision: str = "main"
    dataset: str
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
    def calc_batch_accuracy(self, batch):
        pass

    @abstractmethod
    def calc_batch_perplexity(self, batch):
        pass

    def calc_batch_ptest(self):
        pass

    @abstractmethod
    def score_batch(self, batch):
        pass

    def run(self):
        dataset = self.load_dataset()
        dataset = dataset["test"].map(self.score_batch, batch_size=self.batch_size, batched=True)
        dataset = dataset.map(self.score_batch, batch_size=self.batch_size, batched=True)

        for batch in iter(dataset):
            print(batch)
            print("*****")
