from datetime import datetime
from typing import Any

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aroviq.core.models import Step, Verdict


class Watchtower:
    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self.layout = Layout()

        # Data storage
        self.agent_logs: list[dict[str, Any]] = []
        self.verdicts: list[dict[str, Any]] = []
        self.stats = {
            "total_steps": 0,
            "blocked": 0,
            "cost_est": 0.0
        }

        # Setup Layout
        self.layout.split(
            Layout(name="main", ratio=9),
            Layout(name="footer", ratio=1) # Fixed height ratio might be tricky, usually size is better but ratio works for full screen
        )
        self.layout["main"].split_row(
            Layout(name="agent_stream", ratio=1),
            Layout(name="interceptor_verdicts", ratio=1)
        )

    def render_agent_stream(self) -> Panel:
        """Renders the left panel: Agent Stream."""
        # Create a table for logs
        table = Table(show_header=False, box=None, expand=True)
        table.add_column("Time", style="dim cyan", width=10)
        table.add_column("Content")

        for log in self.agent_logs[-20:]: # Show last 20
            table.add_row(log["time"], log["content"])

        return Panel(
            table,
            title="[bold blue]Agent Stream[/]",
            border_style="blue",
            box=box.ROUNDED
        )

    def render_interceptor_verdicts(self) -> Panel:
        """Renders the right panel: Interceptor Verdicts."""
        table = Table(show_header=False, box=None, expand=True)
        table.add_column("Verdict", width=10)
        table.add_column("Reason")

        for v in self.verdicts[-20:]:
            color = "green" if v["approved"] else "red"
            status = "APPROVED" if v["approved"] else "BLOCKED"
            table.add_row(
                f"[{color}]{status}[/]",
                Text(v["reason"], style=f"dim {color}")
            )

        return Panel(
            table,
            title="[bold red]Interceptor Verdicts (Clean Room)[/]",
            border_style="red",
            box=box.ROUNDED
        )

    def render_footer(self) -> Panel:
        """Renders the footer stats."""
        grid = Table.grid(expand=True)
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="center", ratio=1)

        blocked_pct = 0.0
        if self.stats["total_steps"] > 0:
            blocked_pct = (self.stats["blocked"] / self.stats["total_steps"]) * 100

        grid.add_row(
            f"[bold]Total Steps:[/] {self.stats['total_steps']}",
            f"[bold]Blocked:[/] {blocked_pct:.1f}%",
            f"[bold]Cost Est:[/] ${self.stats['cost_est']:.4f}"
        )

        return Panel(
            grid,
            title="Session Stats",
            border_style="white",
            box=box.ROUNDED
        )

    def update_view(self) -> Layout:
        """Updates the layout with fresh renders."""
        self.layout["agent_stream"].update(self.render_agent_stream())
        self.layout["interceptor_verdicts"].update(self.render_interceptor_verdicts())
        self.layout["footer"].update(self.render_footer())
        return self.layout

    def on_step(self, step: Step):
        """Callback for when a step is received."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        content = f"[{step.step_type.value}] {step.content}"
        # Truncate for display
        if len(content) > 100:
            content = content[:97] + "..."

        self.agent_logs.append({"time": timestamp, "content": content})
        self.stats["total_steps"] += 1

        # Naive cost estimation (simulation)
        # Assuming roughly $0.002 per 1k chars (simulating GPT-4o-mini-like costs for example)
        self.stats["cost_est"] += len(step.content) * (0.002 / 1000)

    def on_verdict(self, verdict: Verdict):
        """Callback for when a verdict is reached."""
        self.verdicts.append({
            "approved": verdict.approved,
            "reason": verdict.reason
        })
        if not verdict.approved:
            self.stats["blocked"] += 1

    def live(self) -> Live:
        """Returns the Live display context manager configured for this Watchtower."""
        return Live(
            get_renderable=self.update_view,
            refresh_per_second=4,
            console=self.console,
            screen=True # Full screen mode
        )
