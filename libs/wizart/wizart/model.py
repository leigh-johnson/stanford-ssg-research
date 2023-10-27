import os
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_PATH = os.env.get(
    "WIZARDLM_13B_PATH", "/mnt/spindle/huggingface/WizardLM-13B-V1.2"
)

tokenizer = AutoTokenizer.from_pretrained("WizardLM/WizardLM-13B-V1.2")
model = AutoModelForCausalLM.from_pretrained("WizardLM/WizardLM-13B-V1.2")
