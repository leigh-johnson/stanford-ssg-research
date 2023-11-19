ILLEGAL_TOKENS = ["---"]


def clean_python_code(row, field: str):
    command = row[field]
    for token in ILLEGAL_TOKENS:
        command = command.replace(token, "")

    row[field] = command
    return row


def run_python_code(row, field: str):
    pass
