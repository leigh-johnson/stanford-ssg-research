from langchain.chains import LLMChain

from llm_programs.tasks.base import BaseTask
from llm_programs.prompts import PromptTemplateType
from llm_programs.prompts.gsm8k import (
    FEW_SHOT_DIRECT_PROMPT_TEMPLATE,
    ZERO_SHOT_DIRECT_PROMPT_TEMPLATE,
)


class Gsm8kTask(BaseTask):
    dataset = "gsm8k"
    revision = "main"

    def prompt_template_cls(self) -> int:
        if self.num_examples == 0:
            return ZERO_SHOT_DIRECT_PROMPT_TEMPLATE()
        elif self.prompt_type == PromptTemplateType.FEW_SHOT_DIRECT:
            return FEW_SHOT_DIRECT_PROMPT_TEMPLATE()


TASK = Gsm8kTask
