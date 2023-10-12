import click

from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from wizart.agent.base import WizartAgent


@click.command()
@click.option(
    "--instruct-model",
    type=click.Choice(
        ["WizardLM/WizardLM-13B-V1.2"],
        case_sensitive=False,
    ),
    default="WizardLM/WizardLM-13B-V1.2",
)
@click.option(
    "--code-model",
    type=click.Choice(
        ["WizardLM/WizardCoder-Python-13B-V1.0"],
        case_sensitive=False,
    ),
    default="WizardLM/WizardCoder-Python-13B-V1.0",
)
@click.option(
    "--math-model",
    type=click.Choice(
        ["WizardLM/WizardMath-13B-V1.0"],
        case_sensitive=False,
    ),
    default="WizardLM/WizardMath-13B-V1.0",
)
@click.option(
    "--prompt-template",
    type=click.Choice(
        ["few_shot_direct", "few_shot_auto_cot", "few_shot_art"], case_sensitive=False
    ),
)
@click.option(
    "--task",
    type=click.Choice(["gsm8k"]),
)
@click.option("--task", type=click.Choice(["gsm8k"], case_sensitive=False))
def main(
    instruct_model: str,
    code_model: str,
    math_model: str,
    prompt_template: str,
    task: str,
):
    """Benchmark WizART against a task"""
    tools = []
    instruct_llm = HuggingFacePipeline.from_model_id(instruct_model, "text-generation")
    agent = WizartAgent(llm=instruct_llm)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True
    )
    agent_executor.run("How many people live in canada as of 2023?")


if __name__ == "__main__":
    main()
