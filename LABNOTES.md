# 2023-11-08

Working hypothesis: For the GSM8K benchmark task, LLAMA-2-7b-chat-hf will perform worse than OpenAI text-davinci-002 as the reasoning engine in an Automatic multi-step reasoning (ART) LLM architechture.

* Implemented accuracy metric

# 2023-10-30

Realized perplexity measurement would naturally decrease for codegen and tool-enhanced prompts, if comparing tool-augmented response with benchmark ground truth.

- [x] Read ["Evaluating Large Language Models Trained on Code"](https://arxiv.org/pdf/2107.03374.pdf) to understand `pass@k` evaluation metric.
- [x] Read ["ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs"] to understand metrics used by [ToolBench](https://openbmb.github.io/ToolBench/). See also: [ToolEval README](https://github.com/OpenBMB/ToolBench/tree/master/toolbench/tooleval)

## Changelog

* Pass command-line arg `--batch-size` to set a batch size (default: 1)

# 2023-10-28

## Changelog

* Renamed `wizart` package to `llm_programs`

# 2023-10-18

Working hypothesis: For the GSM8K benchmark task, WizardLM will perform worse than OpenAI text-davinci-002 as the reasoning engine in an Automatic multi-step reasoning (ART) LLM architechture.

## Current Goals

- [ ] Replicate ART GSM8K experiment run
  - [~] Direct prompt
  - [ ] Auto chain-of-thought prompt
  - [ ] Tool (ART) prompt
- [ ] Measurements
  - [ ] Reasoning engine's perplexity (measures randomness and confidence of the model). 
  
        Perplexity measures the accuracy and similarity of next-word prediction. A high perplexity value indicates the model is "surprised" by next N-token (model predicted a highly dissimilar result). Low perplexity indicates the next word aligns with the model's N-token prediction.
  - [ ] Change in perplexity between prompting strategies (direct prompt, auto CoT, wizART)

## Changelog

* Remove WizardLM, project seems abandoned or about to commercialize: https://github.com/nlpxucan/abcd/issues/218
* Find a new Instruct-trained 7-13B reasoning model, code model, math model.

# 2023-10-11

Working hypothesis: WizardLM will perform worse than OpenAI text-davinci-002 as the reasoning engine in an Automatic multi-step reasoning (ART) LLM architechture.

- [x] WizART package setup
- [x] Read Langchain agent documentation + examples
- [x] Refine hypothesis, define additional measurements that would explain WHY we see a certain result.