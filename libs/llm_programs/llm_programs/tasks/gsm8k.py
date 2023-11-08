from llm_programs.tasks.base import BaseTask


class Gsm8kTask(BaseTask):
    dataset = "gsm8k"
    revision = "main"


TASK = Gsm8kTask
