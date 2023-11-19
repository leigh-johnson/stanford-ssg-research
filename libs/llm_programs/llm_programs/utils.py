import docker

ILLEGAL_TOKENS = ["---"]

DOCKER_TAG = "python:3.11"


def clean_python_code(row, input_column: str = "generated", output_column="generated"):
    command = row[input_column]
    for token in ILLEGAL_TOKENS:
        command = command.replace(token, "")

    row[output_column] = command
    return row


def run_python_code(
    row, input_column: str = "generated", output_column="program_output", error_column="program_error"
):
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
