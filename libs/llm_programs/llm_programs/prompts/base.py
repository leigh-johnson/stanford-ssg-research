from enum import Enum
from langchain.llms.base import BaseLLM


class PromptTemplateType(str, Enum):
    ZERO_SHOT_DIRECT = "zero_shot_direct"
    FEW_SHOT_DIRECT = "few_shot_direct"
    FEW_SHOT_AUTO_COT = "few_shot_auto_cot"
    FEW_SHOT_TOOL = "few_shot_tool"


def is_zero_shot_direct(llm: BaseLLM) -> bool:
    return (
        llm.metadata.get("prompt_template_type", False)
        is PromptTemplateType.ZERO_SHOT_DIRECT
    )
