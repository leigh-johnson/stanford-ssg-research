from llm_programs.tasks.base import BaseTask
import evaluate


class Gsm8kTask(BaseTask):
    dataset = "gsm8k"
    revision = "be45a9e2ae111e0cbfd91a7028f8de6aa80bc9a5"

    def calc_perplexity(self, batch):
        perplexity = evaluate.load("perplexity", module_type="metric")

        batch["perplexity_questions"] = perplexity.compute(
            predictions=batch["question"], model_id=self.instruct_model_id
        )
        batch["perplexity_questions"] = perplexity.compute(
            predictions=batch["answer"], model_id=self.instruct_model_id
        )
        batch["perplexity_questions"] = perplexity.compute(
            predictions=batch["response"], model_id=self.instruct_model_id
        )

        return batch

    def calc_accuracy(self, row):
        results = []

        expected = self.prompt.parse_final_answer(row["answer"])
        final_answer = row[self.prompt_template_type].split("\n")[-1]
        hit = expected in final_answer

        row[f"{self.prompt_template_type.value}__accuracy"] = hit
        return row

    def score_row(self, row):
        llmchain = self.llmchain()
        response = llmchain.invoke(row)
        row[self.prompt_template_type.value] = response
        row = self.calc_accuracy(row)
        return row

    def score(self, dataset):
        if self.batch_size > 1:
            llmchain = self.llmchain()
            results = llmchain.batch(dataset, batch_size=self.batch_size)
            dataset[self.prompt_template_type.value] = results
            dataset = dataset.map(self.calc_accuracy, desc="Calculating Accuracy")
            return dataset
        else:
            return dataset.map(self.score_row, desc="Scoring")


TASK = Gsm8kTask
