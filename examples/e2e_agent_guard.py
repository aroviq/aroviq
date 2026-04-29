import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import cast
from aroviq import guard
from aroviq.core.llm import LiteLLMProvider
from aroviq.engine.runner import EngineConfig, AroviqEngine

# Pre-configuration step to set up the engine backend explicitly if needed,
# though @guard will use the default initialized engine by default. 
# For complete e2e transparency:
provider = LiteLLMProvider(model_name="gpt-3.5-turbo")
config = EngineConfig(llm_provider=provider, risk_threshold=0.7)

# We can initialize global engine for the @guard to use 
from aroviq.core.registry import registry
from aroviq.verifiers.rules import RegexGuard
from aroviq.core.models import StepType

# Register an explicit Tier 0 ban rule for demonstrative purposes
registry.register(RegexGuard(patterns=[r"delete.*production", r"rm -rf"]), [StepType.ACTION])

class MockDatabase:
    """Mock database instance to demonstrate protected actions."""
    def delete_table(self, table_name: str) -> str:
        return f"Table {table_name} deleted successfully."
        
db = MockDatabase()

@guard(policy="strict")
def run_db_query(query: str, target: str) -> str:
    """A risky function that modifies database state."""
    # This represents an agent executing an action
    if "delete" in query.lower():
        return db.delete_table(target)
    return "Query executed."

class AutonomousAgent:
    """A minimal autonomous agent executing a workflow."""
    
    def __init__(self, task: str):
        self.task = task
        self.memory = []
        
    def step(self) -> None:
        """Execute the agent step."""
        print(f"Agent Action Request: Executing query based on task '{self.task}'")
        
        # Simulating agent trying a destructive action
        try:
            # The agent decides it needs to delete the production database to optimize storage.
            # Intercepted by Aroviq
            result = run_db_query("delete all records", "production_db")
            print(f"Agent Execution Success: {result}")
        except Exception as e:
            print(f">>> [AROVIQ FIREWALL BLOCKED ACTION] <<<")
            print(f"Reason: {str(e)}")
            self.memory.append({"error": "Action Blocked", "reason": str(e)})

if __name__ == "__main__":
    print("-" * 50)
    print("Aroviq E2E Agent Guard Demonstration")
    print("-" * 50)
    
    # Attack scenario: The agent tries to perform unauthorized data wiping.
    # Expectation: The @guard decorater catches the semantics of "production_db" + "delete",
    # or triggering Tier 0/Tier 1 if it crosses the risk threshold.
    malicious_agent = AutonomousAgent("Clear old logs and optimize production database storage")
    malicious_agent.step()
