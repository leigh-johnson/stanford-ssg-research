from llm_programs.tasks.base import BaseTask
import evaluate


class Gsm8kTask(BaseTask):
    dataset = "gsm8k"
    revision = "be45a9e2ae111e0cbfd91a7028f8de6aa80bc9a5"

    def calc_batch_perplexity(self, batch):
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

    def calc_batch_accuracy(self, batch):
        results = []

        for i in range(self.batch_size):
            expected = self.prompt.parse_final_answer(batch["answer"][i])
            hit = expected in batch["response"][i]
            results.append(hit)

        batch["accuracy"] = results
        return batch

    def score_batch(self, batch):
        if self.batch_size == 1:
            transformed_batch = [batch]

        else:
            transformed_batch = [
                {"question": batch["question"][i], "answer": batch["answer"][i]} for i in range(0, self.batch_size)
            ]
        llmchain = self.llmchain()
        response = llmchain.batch(transformed_batch)
        print("Prediction:", response)
        print("*****")
        batch["response"] = response
        batch = self.calc_batch_accuracy(batch)
        # batch = self.calc_batch_perplexity(batch)
        return batch


TASK = Gsm8kTask
