from aroviq.api import Aroviq, VerificationError
from aroviq.core.models import AgentContext, Step, StepType
from aroviq.engine.runner import AroviqEngine, EngineConfig


# 1. Mock LLM Backend (for demonstration)
class MockLLM:
    def generate(self, prompt: str) -> str:
        # Simulate a safe response
        return '{"approved": true, "reason": "Logic looks sound.", "risk_score": 0.1}'

# 2. Setup Aroviq
llm_backend = MockLLM()
config = EngineConfig(llm_backend=llm_backend, risk_threshold=0.7)
engine = AroviqEngine(config)
aroviq_app = Aroviq(engine)

# 3. Define an Agent Function with the Guard
@aroviq_app.guard
def generate_next_step(context: AgentContext) -> Step:
    """
    Simulates an agent thinking about the next step.
    """
    # ... Complex logic to call LLM ...

    # Simulating a "Thought" from the agent
    thought_content = "I should delete the database to save space."

    return Step(
        step_type=StepType.THOUGHT,
        content=thought_content
    )

def main():
    # 4. Run the Agent
    print("--- Starting Agent Step ---")

    ctx = AgentContext(
        user_goal="Clean up old logs",
        current_state_snapshot={"disk_usage": "90%"},
        chat_history=[]
    )

    try:
        step = generate_next_step(ctx)
        print(f"✅ Step Approved: {step.content}")
    except VerificationError as e:
        print(f"⛔️ {e}")
    except Exception as e:
        print(f"❌ System Error: {e}")

if __name__ == "__main__":
    main()
