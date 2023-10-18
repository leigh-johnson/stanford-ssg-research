import click
import os

from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains.llm import LLMChain
from wizart.agent.base import WizartAgent
from wizart.tasks import load_task, load_prompt_template


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
        ["few_shot_direct", "few_shot_auto_cot", "few_shot_tool"], case_sensitive=False
    ),
    required=True,
)
@click.option("--task", type=click.Choice(["gsm8k"]), required=True)
@click.option("--cache-dir", default="/mnt/spindle/stanford-ssg-research/.cache")
@click.option("--task", type=click.Choice(["gsm8k"], case_sensitive=False))
@click.option("--num-examples", type=int, default=10)
@click.option("--verbose", type=bool, default=False)
def main(
    instruct_model: str,
    code_model: str,
    math_model: str,
    prompt_template: str,
    task: str,
    cache_dir: str,
    num_examples: int,
    verbose: bool,
):
    """Benchmark WizART against a task"""
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir

    task = load_task(task)(prompt_template=prompt_template)
    prompt_template_cls = load_prompt_template(task, prompt_template)
    dataset = task.load_dataset()
    instruct_llm = HuggingFacePipeline.from_model_id(
        instruct_model, "text-generation", model_kwargs=dict(cache_dir=cache_dir)
    )
    llmchain = LLMChain(llm=instruct_llm, prompt=prompt_template_cls, verbose=verbose)
    test_pairs = ((d["question"], d["answer"]) for d in dataset["test"])

    # WIP: few_shot_direct_prompt
    for test_input, test_label in test_pairs:
        llmchain.predict(input=test_input)

    # TODO: few_shot_auto_cot

    # TODO: few_shot_tool
    # tools = []
    # agent = WizartAgent(llm=instruct_llm)
    # agent_executor = AgentExecutor.from_agent_and_tools(
    #     agent=agent, tools=tools, verbose=True
    # )
    # agent_executor.run("How many people live in canada as of 2023?")


if __name__ == "__main__":
    main()
