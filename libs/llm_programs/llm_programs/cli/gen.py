import click
import os
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline as hf_pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.globals import set_debug, set_verbose

from llm_programs.prompts.base import PromptTemplateType
from llm_programs.tasks import load_task
from llm_programs.llms.huggingface_pipeline import BatchedHuggingFacePipeline


@click.command()
@click.option("--batch-size", type=int, default=1)
@click.option("--cache-dir", default="/mnt/spindle/stanford-ssg-research/.cache")
@click.option(
    "--code-model",
    type=click.Choice(
        ["WizardLM/WizardCoder-Python-13B-V1.0"],
        case_sensitive=False,
    ),
    default="WizardLM/WizardCoder-Python-13B-V1.0",
)
@click.option("--dataset-split", type=str, default="test")
@click.option(
    "--instruct-model",
    type=click.Choice(
        ["meta-llama/Llama-2-7b-chat-hf"],
        case_sensitive=False,
    ),
    default="meta-llama/Llama-2-7b-chat-hf",
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
    "--max-length",
    type=int,
    default=4096,
    help="https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig.max_new_tokens",
)
@click.option(
    "--num-examples",
    type=int,
    default=0,
    help="Number of examples/demos provided in prompt template (few-shot). 0 means no examples will be provided (zero-shot).",
)
@click.option(
    "--num-return-sequences",
    type=int,
    default=1,
    help="The number of highest-scoring beams that should be returned when using beam search, see: https://huggingface.co/blog/how-to-generate",
)
@click.option(
    "--prompt-template",
    type=click.Choice(PromptTemplateType, case_sensitive=False),
    required=True,
)
@click.option(
    "--sample",
    type=bool,
    default=True,
    help="Sample generated token sequences. If false, greedy match is used.",
)
@click.option("--task", type=click.Choice(["gsm8k"], case_sensitive=False))
@click.option(
    "--temperature",
    type=float,
    default=0.6,
    help="The temperature is a parameter that controls the randomness of the LLM's output. A higher temperature will result in more creative and imaginative text, while a lower temperature will result in more accurate and factual text.",
)
@click.option(
    "--top-p",
    type=float,
    default=0.9,
    help="Nucleus sampling threshold",
)
@click.option("--verbose", is_flag=True, default=False)
def main(
    batch_size: int,
    cache_dir: str,
    code_model: str,
    dataset_split: str,
    instruct_model: str,
    math_model: str,
    max_length: int,
    num_examples: int,
    num_return_sequences: int,
    prompt_template: str,
    sample: int,
    task: str,
    temperature: float,
    top_p: float,
    verbose: bool,
):
    """Benchmark llm_programs against a task"""
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir

    now = int(datetime.now().timestamp())
    dataset_outdir = f"{cache_dir}/experiments/{task}_{instruct_model}_{prompt_template.value}_{now}/"

    if verbose:
        set_debug(True)
        set_verbose(True)

    tokenizer = AutoTokenizer.from_pretrained(instruct_model)

    # ref: https://discuss.huggingface.co/t/llama2-pad-token-for-batched-inference/48020
    if batch_size > 1:
        tokenizer.pad_token = tokenizer.bos_token
        tokenizer.padding_side = "left"

    pipeline_kwargs = dict(
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        eos_token_id=tokenizer.eos_token_id,
    )

    model_kwargs = dict(temperature=temperature, top_p=top_p, do_sample=sample)

    pipeline = hf_pipeline(
        task="text-generation",
        model=instruct_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        batch_size=batch_size,
        model_kwargs=model_kwargs,
        tokenizer=tokenizer,
        num_return_sequences=num_return_sequences,
    )

    metadata = dict(num_examples=num_examples, prompt_template_type=prompt_template)

    if batch_size > 1:
        llm = BatchedHuggingFacePipeline(
            pipeline=pipeline,
            model_id=instruct_model,
            batch_size=batch_size,
            pipeline_kwargs=pipeline_kwargs,
            model_kwargs=model_kwargs,
            metadata=metadata,
        )
    else:
        llm = HuggingFacePipeline(
            pipeline=pipeline,
            model_id=instruct_model,
            batch_size=batch_size,
            pipeline_kwargs=pipeline_kwargs,
            model_kwargs=model_kwargs,
            metadata=metadata,
        )
    print(f"Loaded instruct model: {instruct_model}")

    task_kwargs = dict(
        num_examples=num_examples,
        prompt_template_type=prompt_template,
        llm=llm,
        verbose=verbose,
        instruct_model_id=instruct_model,
        batch_size=batch_size,
        dataset_split=dataset_split,
        dataset_outdir=dataset_outdir,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        sample=sample,
    )
    task_runner = load_task(task, task_kwargs)
    task_runner.run()

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
