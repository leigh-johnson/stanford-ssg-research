from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate


DATA = {
    "id": "gsm8k",
    "name": "Middle school arithmetic problems",
    "task_description_with_thoughts": "Answer the following middle school math word problems, which require multi-step arithmetic reasoning.",
    "task_description_with_tools": "(Grade school math) Solve the following middle-school arithmetic problems, writing out intermediate arithmetic calculations as python code. Store your result as a variable named 'ans'.",
    "examples_with_thoughts": [
        {
            "input": "Mason is cleaning out all the junk in his attic. 20% of the items are useful, 10% are valuable heirlooms, and 70% are junk. If Mason's attic has 8 useful items in it, how many junk items does it have?",
            "thoughts": """If Mason had a total of x items in his attic, 20% of the items are useful, 10% are valuable heirlooms, and 70% are junk.
We need to figure out what x is, given that 20% of x is 8.
We can do this by dividing 8 by 0.2 (20%) to get 40. This means that there are 40 items in total the attic.
Finally, to find the number of junk items, we need to figure out what 70% of 40 is. We can do this by multiplying 40 by 0.7 to get 28. 
This means that there are 28 junk items in the attic.""",
            "answer": "28 junk items",
        },
        {
            "input": "A gecko eats 70 crickets every three days.  The first day she eats 30% of the crickets. The second day she eats 6 less than the first, and the third day she finishes up the remaining crickets.  How many crickets does she eat on the third day?",
            "thoughts": """On the first day, the gecko eats 30% of 70 crickets, which is 21 crickets.
On the second day, she eats 6 less than that, so she eats 15 crickets.
On the third day, she eats the remaining crickets. That will be 70 - 21 - 15, which is 34.""",
            "answer": "34 crickets.",
        },
        {
            "input": "My new house has 12 medium ceiling lights but I have seen small and large ceiling lights in other rooms. The small ones require 1 bulb, the medium ones require 2, and the large ones need 3 bulbs. How many bulbs should I buy if my wife says she saw twice as many large ceiling lights as medium ceiling lights and ten more small lights than medium ones?",
            "thoughts": """First, we need to figure out how many large and small ceiling lights there are.
We know that there are 12 medium ceiling lights, so if there are twice as many large ones, that means there are 24 large ones.
We also know that there are ten more small ones than medium ones, so that means there are 22 small ones.
Now that we know how many of each type there are, we can figure out how many bulbs we need.
Remember, small ones require 1 bulb, medium ones require 2, and large ones require 3.
That means we need 22 bulbs for the small ones, 24 bulbs for the medium ones, and 72 bulbs for the large ones.
All together, we need 118 bulbs.""",
            "answer": "118 bulbs",
        },
        {
            "input": "Tim buys a cabinet for $1200 and gets a 15% discount. How much did he pay?",
            "thoughts": "To calculate the discount Tim gets, we find 15%% of 1200. This is 180. Subtracting the discount amount from 1200 gets us 1020. Thus, Tim paid 1020.",
            "answer": "$1020",
        },
        {
            "input": "Grant scored 10 points higher on his math test than John. John received twice as many points as Hunter who scored a 45 on his math test.  What was Grant's test score?",
            "thoughts": "Hunter scored a 45 on his math test. John received twice as many points as Hunter. Thus John got 90. Grant scored 10 points higher on his math test than John. So Grant got a 100 on the test.",
            "answer": "100 points",
        },
    ],
    "examples_with_tools": [
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
Q3: [add unit] Add the appropriate unit to the final answer.
#3: 18 toys
Q3: [EOQ]""",
            "answer": "18 toys",
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
Q4: [EOQ]""",
            "answer": "270 miles",
        },
    ],
}


EXAMPLE_TOOL_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["input", "actions", "answer"],
    template="Input: {input}\n{actions}\nFinal Answer: {answer}",
)

FEW_SHOT_TOOL_PROMPT_TEMPLATE = FewShotPromptTemplate(
    examples=DATA["examples_with_tools"],
    example_prompt=EXAMPLE_TOOL_PROMPT_TEMPLATE,
    prefix=DATA["task_description_with_tools"],
    suffix="Question: {input}",
    input_variables=["input"],
)

EXAMPLE_COT_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["input", "thoughts", "answer", "cot_prompt"],
    template="Input: {input}\n Answer: Let's think step-by-step.\n{thoughts}\nFinal Answer: {answer}",
)

FEW_SHOT_COT_PROMPT_TEMPLATE = FewShotPromptTemplate(
    examples=DATA["examples_with_thoughts"],
    example_prompt=EXAMPLE_COT_PROMPT_TEMPLATE,
    prefix=DATA["task_description_with_thoughts"],
    suffix="Question: {input}",
    input_variables=["input"],
)

EXAMPLE_DIRECT_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["input", "answer"],
    template="Input: {input}\n Final Answer: {answer}",
)

FEW_SHOT_DIRECT_PROMPT_TEMPLATE = FewShotPromptTemplate(
    examples=DATA["examples_with_thoughts"],
    example_prompt=EXAMPLE_DIRECT_PROMPT_TEMPLATE,
    prefix=DATA["task_description_with_thoughts"],
    suffix="Question: {input}",
    input_variables=["input"],
)
