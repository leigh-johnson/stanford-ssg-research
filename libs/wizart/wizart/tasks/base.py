from enum import Enum
import datasets

from langchain.pydantic_v1 import BaseModel
from langchain.prompts.few_shot import FewShotPromptTemplate


class PromptTemplateType(Enum):
    FEW_SHOT_DIRECT = "few_shot_direct"
    FEW_SHOT_AUTO_COT = "few_shot_auto_cot"
    FEW_SHOT_TOOL = "few_shot_tool"


class BaseTask(BaseModel):
    dataset: str
    streaming: bool = True
    revision: str = "main"

    prompt_template: PromptTemplateType

    def load_dataset(self):
        return datasets.load_dataset(
            self.dataset,
            self.revision,
            streaming=self.streaming,
        )
