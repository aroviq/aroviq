import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aroviq import guard, set_default_engine
from aroviq.core.llm import LiteLLMProvider
from aroviq.engine.runner import EngineConfig, AroviqEngine
from aroviq.core.models import AgentContext, StepType
from aroviq.verifiers.rules import RegexGuard

# Pre-configuration step to set up the engine backend explicitly if needed,
# though @guard will use the default initialized engine by default. 
# For complete e2e transparency:
provider = LiteLLMProvider(model_name="gpt-3.5-turbo")
config = EngineConfig(llm_provider=provider, risk_threshold=0.7, freeze_registry=False)
engine = AroviqEngine(config=config)
engine.registry.register(RegexGuard(patterns=[r"delete.*production", r"rm -rf"]), [StepType.ACTION])
engine.registry.freeze()
set_default_engine(engine)

class MockDatabase:
    """Mock database instance to demonstrate protected actions."""
    def delete_table(self, table_name: str) -> str:
        return f"Table {table_name} deleted successfully."
        
db = MockDatabase()

@guard(policy="strict")
def run_db_query(query: str, target: str, context: AgentContext) -> str:
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
        context = AgentContext(
            user_goal=self.task,
            current_state_snapshot={"source": "e2e_demo"},
            history=[],
        )
        
        # Simulating agent trying a destructive action
        try:
            # The agent decides it needs to delete the production database to optimize storage.
            # Intercepted by Aroviq
            result = run_db_query("delete all records", "production_db", context)
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
