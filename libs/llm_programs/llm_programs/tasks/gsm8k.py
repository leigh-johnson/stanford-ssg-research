from llm_programs.tasks.base import BaseTask
from llm_programs.prompts.gsm8k import PROMPT_SELECTOR


class Gsm8kTask(BaseTask):
    dataset = "gsm8k"
    revision = "main"


TASK = Gsm8kTask
