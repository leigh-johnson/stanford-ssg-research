import docker
import re

DOCKER_TAG = "python:3.11"


def extract_python_code(row, input_column: str = "generated", output_column="program", regex=r"```$(.*)```"):
    """
    Extracts python code between separator tokens: ```
    """

    match = re.match(regex, row[input_column])
    if match:
        row[output_column] = match.group(0)
    else:
        row[output_column] = match
    return row


def run_python_code(row, input_column: str = "program", output_column="program_output", error_column="program_error"):
    command = ["python3", "-c", row[input_column]]
    client = docker.from_env()
    print("Running command: \n", command)
    try:
        result = client.containers.run(DOCKER_TAG, command=command, remove=True)
        row[output_column] = result
        row[error_column] = False
    except docker.errors.ContainerError as e:
        row[output_column] = str(e)
        row[error_column] = True
    return row
