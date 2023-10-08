import click


@click.command()
@click.option("--model", type=click.Choice(["wizardlm-13b"], case_sensitive=False))
@click.option(
    "--strategy",
    type=click.Choice(["few_shot_direct", "few_shot_aut_cot", "few_shot_art"]),
)
@click.option("--task", type=click.Choice(["gsm8k"], case_sensitive=False))
def main(model: str, strategy: str, task: str):
    """Benchmark WizART against a task"""
    pass


if __name__ == "__main__":
    main()
