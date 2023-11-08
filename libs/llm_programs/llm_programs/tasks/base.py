from abc import ABC, abstractmethod
import datasets
from datasets import Dataset

from langchain.pydantic_v1 import BaseModel
from langchain.schema.language_model import BaseLanguageModel
from llm_programs.prompts.base import PromptTemplateType, BasePromptSelector
from langchain.chains.prompt_selector import ConditionalPromptSelector


class BaseTask(BaseModel, ABC):
    dataset: str
    dataset_revision: str = "main"
    llm: BaseLanguageModel
    num_examples: int = 0
    prompt_template_type: PromptTemplateType
    prompt_selector: BasePromptSelector
    streaming: bool = True
    verbose: bool = False
    batch_size: int = 1

    def llmchain(self):
        prompt = self.prompt_selector.get_prompt()
        return prompt | self.llm

    def load_dataset(self):
        return datasets.load_dataset(
            self.dataset,
            self.dataset_revision,
            streaming=self.streaming,
        )

    def calc_batch_accuracy(self, input_batch: Dataset, ground_truth_batch: Dataset):
        pass

    def calc_batch_perplexity(self):
        pass

    def calc_batch_ptest(self):
        pass

    def score_batch(self, batch):
        if self.batch_size == 1:
            batch = [batch]

        else:
            batch = [
                {"question": batch["question"][i], "answer": batch["answer"][i]}
                for i in range(0, self.batch_size)
            ]
        llmchain = self.llmchain()
        responses = Dataset.from_list(llmchain.batch(batch))
        print("Prediction:", responses)
        print("*****")

    def run(self):
        dataset = self.load_dataset()
        dataset["test"].map(self.score_batch, batch_size=self.batch_size)
