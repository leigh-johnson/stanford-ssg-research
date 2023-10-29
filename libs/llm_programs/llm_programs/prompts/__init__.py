from enum import Enum


class PromptTemplateType(str, Enum):
    ZERO_SHOT_DIRECT = "zero_shot_direct"
    FEW_SHOT_DIRECT = "few_shot_direct"
    FEW_SHOT_AUTO_COT = "few_shot_auto_cot"
    FEW_SHOT_TOOL = "few_shot_tool"
