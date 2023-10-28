from typing import List, Optional

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools.python.tool import PythonREPLTool
from langchain.tools import BaseTool
from langchain.schema.language_model import BaseLanguageModel
from langchain.pydantic_v1 import BaseModel, Field
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain


class Basellm_programsTool(BaseModel):
    llm: BaseLanguageModel = Field(
        default_factory=lambda: HuggingFacePipeline.from_model_id(
            model_id="WizardLM/WizardLM-13B-V1.2", task="text-generation"
        )
    )
    template: PromptTemplate


class WizardLLMTool(Basellm_programsTool, BaseTool):
    @staticmethod
    def get_description(name, description):
        raise NotImplemented
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None,)
        llm_chain = LLMChain.from_string(llm=self.llm, template=self.template)
        output = llm_chain.run(input=query)
        return output

class llm_programsToolkit(BaseToolkit):
    def get_tools(self) -> List[BaseTool]:
        return [
            PythonREPLTool,
            WizardLLMTool
        ]
