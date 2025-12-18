#!/usr/bin/env python3
import fire

from aroviq.cli.report import generate_report


class CLI:
    def report(self, file: str = "aroviq_trace.jsonl"):
        """Generates a safety report from a log file."""
        generate_report(file)

    def watch(self):
        """Launches the verifiable watchtower (TUI)."""
        # This is just a placeholder to show we can group commands.
        # Watchtower usage is typically programmatic but could be a distinct CLI mode if it hooked into a running process.
        print("Watchtower is designed to be used programmatically in your agent loop.")

def main():
    fire.Fire(CLI)

if __name__ == "__main__":
    main()
