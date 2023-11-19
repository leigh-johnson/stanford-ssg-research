from datasets import Dataset
from llm_programs.tasks.base import BaseTask
from llm_programs.prompts.base import PromptTemplateType
from llm_programs.utils import clean_python_code, run_python_code
import evaluate


class Gsm8kTask(BaseTask):
    dataset = "gsm8k"
    dataset_revision = "main"

    generated_column = "generated"
    program_output_column = "program_output"
    program_error_column = "program_error"

    def calc_perplexity(self, batch):
        perplexity = evaluate.load("perplexity", module_type="metric")

        batch["perplexity_questions"] = perplexity.compute(predictions=batch["question"], model_id=self.instruct_model)
        batch["perplexity_questions"] = perplexity.compute(predictions=batch["answer"], model_id=self.instruct_model)
        batch["perplexity_questions"] = perplexity.compute(predictions=batch["response"], model_id=self.instruct_model)

        return batch

    def calc_accuracy(self, row):
        expected = self.prompt.parse_final_answer(row["answer"])
        final_answer = row[self.generated_column].split("\n")[-1]
        hit = expected in final_answer

        row["accuracy"] = hit
        return row

    def score_language_row(self, row):
        llmchain = self.llmchain()
        response = llmchain.invoke(row)
        row[self.generated_column] = response
        row = self.calc_accuracy(row)
        return row

    def score_program_row(self, row):
        llmchain = self.llmchain()
        response = llmchain.invoke(row)
        row[self.generated_column] = response
        row = clean_python_code(row)
        row = run_python_code(row)
        return row

    def score_language_output(self, dataset) -> Dataset:
        if self.batch_size > 1:
            llmchain = self.llmchain()
            results = llmchain.batch(dataset, batch_size=self.batch_size)
            dataset = dataset.add_column(self.generated_column, results)
            return dataset.map(self.calc_accuracy, relesc="Calculating Accuracy")
        return dataset.map(self.score_language_row, desc="Scoring")

    def score_program_output(self, dataset) -> Dataset:
        if self.batch_size > 1:
            llmchain = self.llmchain()
            results = llmchain.batch(dataset, batch_size=self.batch_size)
            dataset = dataset.add_column(self.generated_column, results)
            dataset = dataset.map(clean_python_code)
            dataset = dataset.map(run_python_code)
            return dataset
        else:
            return dataset.map(self.score_program_row, desc="Scoring")

    def score(self, dataset) -> Dataset:
        if self.prompt_template_type is PromptTemplateType.PROGRAM:
            return self.score_program_output(dataset)
        elif self.prompt_template_type in [PromptTemplateType.DIRECT, PromptTemplateType.COT]:
            return self.score_language_output(dataset)


TASK = Gsm8kTask
