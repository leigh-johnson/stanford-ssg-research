from pydantic import BaseModel
from typing import List
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate


class BasePromptModel(BaseModel):
    id: str
    name: str
    description: str
    examples: List[str]
    example_prompt_template: PromptTemplate
    few_shot_prompt_template: FewShotPromptTemplate
