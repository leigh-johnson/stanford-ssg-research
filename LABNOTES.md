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