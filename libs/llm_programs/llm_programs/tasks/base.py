from abc import ABC, abstractmethod
import datasets
import evaluate

from datasets import Dataset

from langchain.pydantic_v1 import BaseModel
from langchain.schema.language_model import BaseLanguageModel
from llm_programs.prompts.base import PromptTemplateType, BasePrompt


class BaseTask(BaseModel, ABC):
    dataset: str
    dataset_revision: str = "main"
    llm: BaseLanguageModel
    num_examples: int = 0
    prompt_template_type: PromptTemplateType
    prompt: BasePrompt
    streaming: bool = True
    verbose: bool = False
    batch_size: int = 1

    def llmchain(self):
        prompt = self.prompt.get_prompt()
        return prompt | self.llm

    def load_dataset(self):
        return datasets.load_dataset(
            self.dataset,
            self.dataset_revision,
            streaming=self.streaming,
        )

    def calc_batch_accuracy(self, batch):
        results = []

        for i in range(self.batch_size):
            expected = self.prompt.parse_final_answer(batch["answer"][i])
            hit = expected in batch["responses"][i]
            results.append(hit)

        batch["results"] = results
        return batch

    def calc_batch_perplexity(self):
        pass

    def calc_batch_ptest(self):
        pass

    def score_batch(self, batch):
        if self.batch_size == 1:
            transformed_batch = [batch]

        else:
            transformed_batch = [
                {"question": batch["question"][i], "answer": batch["answer"][i]} for i in range(0, self.batch_size)
            ]
        llmchain = self.llmchain()
        responses = llmchain.batch(transformed_batch)
        print("Prediction:", responses)
        print("*****")
        batch["responses"] = responses
        return self.calc_batch_accuracy(batch)

    def run(self):
        dataset = self.load_dataset()
        dataset = dataset["test"].map(self.score_batch, batch_size=self.batch_size, batched=True)
        dataset = dataset.map(self.score_batch, batch_size=self.batch_size, batched=True)
        next(iter(dataset))
        next(iter(dataset))
