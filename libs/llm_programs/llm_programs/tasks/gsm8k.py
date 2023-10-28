from langchain.chains import LLMChain

from llm_programs.tasks.base import BaseTask
from llm_programs.prompts.gsm8k import FEW_SHOT_DIRECT_PROMPT_TEMPLATE


class Gsm8kTask(BaseTask):
    dataset = "gsm8k"
    revision = "main"


class DirectPromptChain(LLMChain):
    prompt = FEW_SHOT_DIRECT_PROMPT_TEMPLATE


TASK = Gsm8kTask
