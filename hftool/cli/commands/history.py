"""History command — view and manage command history."""

import sys
from typing import Optional

import click


@click.command("history")
@click.option("--clear", is_flag=True, help="Clear all history")
@click.option("--rerun", type=int, metavar="ID", help="Re-run command from history")
@click.option("--limit", "-n", type=int, default=10, help="Number of entries to show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def history_command(
    ctx: click.Context,
    clear: bool,
    rerun: Optional[int],
    limit: int,
    as_json: bool,
):
    """View and manage command history.

    \b
    Examples:
        hftool history                 # Show recent history
        hftool history -n 20           # Show last 20 commands
        hftool history --rerun 42      # Re-run command #42
        hftool history --clear         # Clear all history
    """
    from hftool.core.history import History

    history = History.get()

    # Clear history
    if clear:
        if click.confirm("Clear all command history?"):
            history.clear()
            click.echo("History cleared.")
        return

    # Re-run command
    if rerun is not None:
        from hftool.cli.commands.task import run_task_command

        entry = history.get_by_id(rerun)
        if entry is None:
            click.echo(f"Error: No history entry with ID {rerun}", err=True)
            sys.exit(1)

        click.echo(f"Re-running command #{entry.id} from {entry.get_timestamp_str()}:")
        click.echo(f"  {entry.to_command()}")
        click.echo("")

        if not click.confirm("Continue?", default=True):
            return

        # Extract parameters and re-run
        run_task_command(
            ctx=ctx,
            task=entry.task,
            model=entry.model,
            input_data=entry.input_data,
            output_file=entry.output_file,
            device=entry.device,
            dtype=entry.dtype,
            seed=entry.seed,
            interactive=False,
            verbose=ctx.obj.get("verbose", False),
            open_output=ctx.obj.get("open"),
        )
        return

    # Show history
    entries = history.get_recent(limit=limit)

    if not entries:
        click.echo("No command history yet.")
        return

    if as_json:
        import json
        from dataclasses import asdict
        output = [asdict(entry) for entry in entries]
        click.echo(json.dumps(output, indent=2))
        return

    # Text output
    click.echo("")
    click.echo("Recent command history:")
    click.echo("=" * 80)

    for entry in entries:
        # Status indicator
        status = click.style("✓", fg="green") if entry.success else click.style("✗", fg="red")

        # Header
        click.echo(f"\n[{entry.id}] {status} {entry.get_timestamp_str()} - {entry.task}")

        # Details
        if entry.model:
            click.echo(f"    Model: {entry.model}")

        # Show input (truncate if too long)
        input_display = entry.input_data
        if len(input_display) > 60:
            input_display = input_display[:57] + "..."
        click.echo(f"    Input: {input_display}")

        if entry.output_file:
            click.echo(f"    Output: {entry.output_file}")

        if entry.seed is not None:
            click.echo(f"    Seed: {entry.seed}")

        if not entry.success and entry.error_message:
            error_display = entry.error_message
            if len(error_display) > 60:
                error_display = error_display[:57] + "..."
            click.echo(click.style(f"    Error: {error_display}", fg="red"))

        # Show command for reproduction
        click.echo(click.style(f"    Rerun: hftool history --rerun {entry.id}", fg="cyan"))

    click.echo("")
    click.echo("=" * 80)
    click.echo(f"Showing {len(entries)} most recent commands")
