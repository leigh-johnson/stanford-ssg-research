from llm_programs.tasks.base import BaseTask, PromptTemplateType
from llm_programs.prompts.gsm8k import PROMPT_SELECTOR, DIRECT_PROMPT_TASK_DESCRIPTION


class Gsm8kTask(BaseTask):
    dataset = "gsm8k"
    revision = "main"

    prompt_selector = PROMPT_SELECTOR

    def task_description(self) -> str:
        if self.prompt_template_type is PromptTemplateType.ZERO_SHOT_DIRECT:
            return
        raise NotImplementedError(
            f"Task description for {self.prompt_template_type} is not yet implemented, please add to promots/gsm8k.py"
        )


TASK = Gsm8kTask
