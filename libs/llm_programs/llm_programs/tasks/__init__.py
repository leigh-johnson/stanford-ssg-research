import importlib
from llm_programs.tasks.base import BaseTask
from langchain.prompts.few_shot import FewShotPromptTemplate


def load_task(task: str) -> BaseTask:
    module = importlib.import_module(f"llm_programs.tasks.{task}")
    return module.TASK


def load_prompt_template(task: str, prompt_type: str) -> FewShotPromptTemplate:
    module = importlib.import_module(f"llm_programs.tasks.{task}")
    if prompt_type == "few_shot_direct":
        return module.FEW_SHOT_DIRECT_PROMPT_TEMPLATE
    # elif prompt_type == "few_shot_auto_cot":
    # elif prompt_type == "few_shot_tool":
    raise NotImplemented
