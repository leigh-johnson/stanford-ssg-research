from langchain.agents.agent import Agent, AgentExecutor, AgentOutputParser
from langchain.agents.agent_types import AgentType
from langchain.pydantic_v2 import Field
from .parser import WizARTOutputParser


class WizARTAgent(Agent):
    """Agent for WizART chain"""

    # TODO
    # output_parser: AgentOutputParser = Field(default_factory=WizARTOutputParser)

    @property
    def _agent_type(self) -> str:
        """Return Identifier of an agent type."""
        return "wizart"
