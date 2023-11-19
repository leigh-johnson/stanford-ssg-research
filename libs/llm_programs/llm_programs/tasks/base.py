import json
import os
from abc import ABC, abstractmethod
import datasets

from langchain.pydantic_v1 import BaseModel, Field
from pydantic.v1.json import pydantic_encoder

from langchain.schema.language_model import BaseLanguageModel
from llm_programs.prompts.base import PromptTemplateType, BasePrompt


class BaseTask(BaseModel, ABC):
    batch_size: int
    dataset_revision: str
    dataset: str
    dataset_split: str
    dataset_outdir: str
    instruct_model_id: str
    llm: BaseLanguageModel = Field(exclude=True)  # exclude from serialization
    max_length: int
    num_examples: int
    num_return_sequences: int
    prompt_template_type: PromptTemplateType
    prompt: BasePrompt
    sample: bool
    streaming: bool = False
    temperature: float
    top_p: float
    verbose: bool = False

    def llmchain(self):
        prompt = self.prompt.get_prompt()
        return prompt | self.llm

    def load_dataset(self):
        return datasets.load_dataset(
            self.dataset,
            self.dataset_revision,
            streaming=self.streaming,
            split=self.dataset_split,
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

    def save_params(self):
        filename = os.path.join(self.dataset_outdir, "params.json")
        with open(filename, "w+", encoding="utf-8") as f:
            json.dump(self, f, indent=4, default=pydantic_encoder)
        print(f"Wrote params to {filename}")

    def run(self):
        dataset = self.load_dataset()
        dataset = self.score(dataset)
        # dataset = dataset.map(self.score, desc="Scoring")
        # for item in dataset:
        #     import pdb
        #     print(item)
        dataset.save_to_disk(self.dataset_outdir)
        self.save_params()
