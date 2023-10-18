from langchain.chains import LLMChain

from wizart.tasks.base import BaseTask
from wizart.prompts.gsm8k import FEW_SHOT_DIRECT_PROMPT_TEMPLATE


class Gsm8kTask(BaseTask):
    dataset = "gsm8k"
    revision = "main"


class DirectPromptChain(LLMChain):
    prompt = FEW_SHOT_DIRECT_PROMPT_TEMPLATE


TASK = Gsm8kTask
