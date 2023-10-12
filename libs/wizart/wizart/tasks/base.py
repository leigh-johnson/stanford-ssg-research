from dataclasses import dataclass
from enum import Enum
from langchain.prompts.few_shot import FewShotPromptTemplate


class PromptTemplateType(Enum):
    FEW_SHOT_DIRECT = "few_shot_direct"
    FEW_SHOT_AUTO_COT = "few_shot_auto_cot"
    FEW_SHOT_ART = "few_shot_art"


class TaskType(Enum):
    GSM8K = "gsm8k"


class ModelType(Enum):
    WIZARDLM_13B_V12 = "WizardLM/WizardLM-13B-V1.2"


@dataclass
class BaseTaskRunner:
    strategy: PromptTemplateType
    template: FewShotPromptTemplate
    task: TaskType
    model: ModelType

    def run(self):
        raise NotImplemented
