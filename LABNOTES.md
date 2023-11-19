
# 2023-11-15

Comparing perplexity within easy, medium, hard tasks.

"Easy" meaning a task the model is well-suited for.
"Medium" some middle-of-the-road task
"Hard" some task the model is not well-suited for.

Relationship between difficulty and perplexity score, thinking about some metric in terms of ratio of perplexity : of successful easy task to perplexity : of successful medium task.

# 2023-11-12

* Test different decoding strategies: 
  * Greedy
  * Beam Search
  * Random Sampling
  * Random Sampling with Temperature
  * Top K Sampling
  * Nucleaus Sampling

# 2023-11-08

Working hypothesis: For the GSM8K benchmark task, LLAMA-2-7b-chat-hf will perform worse than OpenAI text-davinci-002 as the reasoning engine in an Automatic multi-step reasoning (ART) LLM architechture.

Presentation next week:
* What would I learn if the hypothesis is true?
* What would I learn if the hypothesis is false?
* What am I measuring to explain the results observed in my experiment?
  * Perplexity
  * Performance on base GSM8K vs. performance on new problems of my design

* Can we use an annealed metric to see how changes affect the final results? 
  * Start with a high degree of change in temperature, then decrease

* Create a hold-out set of GSM8K prompts
* What is Davinci's dataset cut-off date? What is LLAMA-2's cutoff date?

* Are there other modalities we'd want to test? e.g. physics with a string tied to a nail.

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