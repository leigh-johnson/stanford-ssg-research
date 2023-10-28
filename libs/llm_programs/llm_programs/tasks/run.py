import click
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline as hf_pipeline
from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains.llm import LLMChain

from llm_programs.agent.base import llm_programsAgent
from llm_programs.tasks import load_task, load_prompt_template


@click.command()
@click.option(
    "--instruct-model",
    type=click.Choice(
        ["meta-llama/Llama-2-7b-hf"],
        case_sensitive=False,
    ),
    default="meta-llama/Llama-2-7b-hf",
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
@click.option(
    "--max-length",
    type=int,
    default=512,
    help="https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.max_new_tokens",
)
@click.option(
    "--cuda-device-id", type=int, default=1, help="CUDA device id, or -1 to use CPU"
)
@click.option("--batch-size", type=int, default=1)
def main(
    instruct_model: str,
    code_model: str,
    math_model: str,
    prompt_template: str,
    task: str,
    cache_dir: str,
    num_examples: int,
    verbose: bool,
    max_length: int,
    cuda_device_id: int,
    batch_size: int,
):
    """Benchmark llm_programs against a task"""
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    task_runner = load_task(task)(prompt_template=prompt_template)
    dataset = task_runner.load_dataset()

    import pdb

    pdb.set_trace()
    prompt_template_cls = load_prompt_template(task, prompt_template)
    model_kwargs = dict(cache_dir=cache_dir, device_map="auto")
    pipeline_kwargs = dict(max_length=max_length)

    tokenizer = AutoTokenizer.from_pretrained(instruct_model, **model_kwargs)
    model = AutoModelForCausalLM.from_pretrained(instruct_model, **model_kwargs)

    pipeline = hf_pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        model_kwargs=model_kwargs,
        **pipeline_kwargs,
    )

    instruct_llm = HuggingFacePipeline(
        pipeline=pipeline,
        model_id=instruct_model,
        model_kwargs=model_kwargs,
        pipeline_kwargs=pipeline_kwargs,
        batch_size=batch_size,
    )
    print(f"Loaded instruct model: {instruct_model}")
    llmchain = LLMChain(llm=instruct_llm, prompt=prompt_template_cls, verbose=verbose)

    for d in dataset["test"]:
        print("Question: ", d["question"])
        print("Expected Answer: ", d["answer"])
        test_input = d["question"]
        # test_answer = d["answer"]
        result = llmchain.predict(input=test_input)
        print("Prediction: ", result)

    # TODO: few_shot_auto_cot

    # TODO: few_shot_tool
    # tools = []
    # agent = llm_programsAgent(llm=instruct_llm)
    # agent_executor = AgentExecutor.from_agent_and_tools(
    #     agent=agent, tools=tools, verbose=True
    # )
    # agent_executor.run("How many people live in canada as of 2023?")


if __name__ == "__main__":
    main()
