from dataclasses import dataclass
from typing import List, Any, Tuple

from langchain.agents.agent import (
    Agent,
    AgentExecutor,
    AgentOutputParser,
    BaseMultiActionAgent,
)

from langchain.schema import (
    AgentAction,
    AgentFinish,
    BaseOutputParser,
    BasePromptTemplate,
    OutputParserException,
)
from langchain.callbacks.manager import (
    Callbacks,
)
from langchain.schema.language_model import BaseLanguageModel
from langchain.chains.llm import LLMChain
from langchain.agents.agent_types import AgentType
from langchain.pydantic_v1 import Field
from .parser import llm_programsOutputParser


@dataclass
class WizardAgentConfig:
    instruct_model: str
    code_model: str
    math_model: str
    prompt_template: str
    task: str


class llm_programsAgent(BaseMultiActionAgent):

    """Agent for llm_programs chain

    Execute multiple tasks in a series, generating and executing Python code for mathmatical reasoning problems.

    Input: A toy manufacturer receives an order for 400 toys. 5 workers are available to work on the order. 2 of the workers produce 6 toys an hour, and another 2 workers produce 4 toys an hour. They all work on the order during their 10-hour shift, and by the end of their shift the manufacturer still needs another 20 toys to be able to ship the order. How many toys per hour does the fifth worker produce?

    Q1: [generate python code] Write down the arithmetic or algebra equations as python code.
    A1: num_toys_ordered = 400
        num_workers = 5
        toys_produced_per_hour_by_worker1 = 6
        toys_produced_per_hour_by_worker2 = 6
        toys_produced_per_hour_by_worker3 = 4
        toys_produced_per_hour_by_worker4 = 4
        toys_produced_per_hour_by_worker5 = Symbol('toys_produced_per_hour_by_worker5', positive=True)
        hours_worked = 10
        toys_produced = num_toys_ordered-20
        toys_produced_by_all_workers = ( toys_produced_per_hour_by_worker1 + toys_produced_per_hour_by_worker2 + toys_produced_per_hour_by_worker3 + toys_produced_per_hour_by_worker4 + toys_produced_per_hour_by_worker5) * hours_worked
        solution = solve_it(toys_produced_by_all_workers - toys_produced, toys_produced_per_hour_by_worker5)
        ans = solution[toys_produced_per_hour_by_worker5]
        print(ans)

    Q2: [code execute] Execute the python code in A1 and get the value of "ans"
    A2: 18

    Q3: [add unit] Add the appropriate unit to the final answer.
    A3: 18 toys

    Final Answer: 18 toys
    """

    llm: BaseLanguageModel

    # TODO
    # output_parser: AgentOutputParser = Field(default_factory=llm_programsOutputParser)

    @property
    def _agent_type(self) -> str:
        """Return Identifier of an agent type."""
        return "llm_programs"

    @property
    def input_keys(self):
        return ["input"]

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ):
        pass

    def aplan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ):
        raise NotImplemented
