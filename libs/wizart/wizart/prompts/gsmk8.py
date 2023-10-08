from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from .base import BasePromptModel

DATA = {
    "id": "gsm8k",
    "name": "Middle school arithmetic problems",
    "description": "Solve the following middle-school arithmetic problems, writing out intermediate arithmetic calculations as python code. Store your result as a variable named 'ans'.",
    "examples": [
        {
            "input": "A toy manufacturer receives an order for 400 toys. 5 workers are available to work on the order. 2 of the workers produce 6 toys an hour, and another 2 workers produce 4 toys an hour. They all work on the order during their 10-hour shift, and by the end of their shift the manufacturer still needs another 20 toys to be able to ship the order. How many toys per hour does the fifth worker produce?",
            "actions": """Q1: [generate python code] write down the arithmetic or algebra equations as python code
#1:
num_toys_ordered = 400
num_workers = 5
toys_produced_per_hour_by_worker1 = 6
toys_produced_per_hour_by_worker2 = 6
toys_produced_per_hour_by_worker3 = 4
toys_produced_per_hour_by_worker4 = 4
toys_produced_per_hour_by_worker5 = Symbol('toys_produced_per_hour_by_worker5', positive=True)
hours_worked = 10
toys_produced = num_toys_ordered-20
toys_produced_by_all_workers = ( toys_produced_per_hour_by_worker1 + toys_produced_per_hour_by_worker2 + toys_produced_per_hour_by_worker3 + toys_produced_per_hour_by_worker4 + toys_produced_per_hour_by_worker5) * hours_worked
solution = solve_it(toys_produced_by_all_workers - toys_produced, toys_produced_per_hour_by_worker5)
ans = solution[toys_produced_per_hour_by_worker5]
print(ans)
Q2: [code execute] Execute the python code in #1 and get the value of "ans"
#2: 18
Q3: [EOQ]
Ans: 18
""",
        },
        {
            "input": "If two trains depart from a station in opposite directions, and one train is traveling 60 miles an hour while the other is traveling half that distance per hour, how far apart are they from each other after 3 hours?",
            "actions": """Q1: [generate python code] write down the arithmetic or algebra equations as python code
#1:
speed_of_first_train = 60
speed_of_second_train = 30
distance_apart = speed_of_first_train * 3 + speed_of_second_train * 3
ans = distance_apart
print(ans)
Q2: [code execute] Execute the python code and get the value of "ans"
#2: 270
Q3: [add unit] Add the appropriate unit to the final answer.
#3: 270 miles
Q4: [EOQ]
Ans: 270 miles""",
        },
    ],
}


EXAMPLE_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["input", "actions"], template="Input: {input}\n{actions}"
)

FEW_SHOT_PROMPT_TEMPLATE = FewShotPromptTemplate(
    examples=DATA["examples"],
    example_prompt=EXAMPLE_PROMPT_TEMPLATE,
    suffix="Question: {input}",
    input_variables=["input"],
)

Gsmk8Prompts = BasePromptModel(
    example_prompt=EXAMPLE_PROMPT_TEMPLATE,
    few_shot_prompt_template=FEW_SHOT_PROMPT_TEMPLATE**DATA,
)
