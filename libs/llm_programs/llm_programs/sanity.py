## this is just a sanity-check for basic pipeline operations

import os
from transformers import AutoTokenizer
import transformers
import torch
import datasets

from langchain.chains import LLMChain
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts.prompt import PromptTemplate

from langchain.globals import set_debug, set_verbose

set_debug(True)
set_verbose(True)

from llm_programs.prompts.gsm8k import ZERO_SHOT_DIRECT_PROMPT_TEMPLATE

CACHE_DIR = "/mnt/spindle/stanford-ssg-research/.cache"

os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)


task_description = "Answer the following middle school math word problem, which requires multi-step arithmetic reasoning."

llm = HuggingFacePipeline(
    pipeline=pipeline,
    model_id=model,
    batch_size=1,
    pipeline_kwargs=dict(
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=512,
    ),
)

prompt_template = "{task_description}\n{question}\n"

dataset = datasets.load_dataset("gsm8k", "main", streaming=True)

# result = llm.predict(prompt)
# TODO cannot re-use instances of PromptTemplate
prompt_template = PromptTemplate(
    input_variables=["question", "task_description"],
    template="""{task_description}
Question: {question}
Answer:
""",
    validate_template=True,
)
# chain = prompt_template | llm.bind(stop="Question:")

for d in dataset["test"]:
    question = d["question"].strip()
    print("Question: \n", question)
    print("Expected Answer: \n", d["answer"])
    chain = prompt_template | llm
    result = chain.invoke({"question": question, "task_description": task_description})
    print(f"Result: {result}")
    print("*********")
# sequences = llm.generate([prompt])

# sequences = pipeline(
#     prompt,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     max_length=512,
# )
# for seq in sequences:
#     print(f"Result: {seq}")
