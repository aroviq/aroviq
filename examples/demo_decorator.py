import aroviq
from aroviq.engine.runner import AroviqEngine, EngineConfig
from aroviq.core.llm import MockProvider
from aroviq.core.models import Step, AgentContext, Verdict, StepType
from aroviq.core.exceptions import SecurityException
from aroviq.core.registry import registry
from aroviq.integrations.decorators import aroviq_guard

# --- 1. Define a simple Keyword Blocker Verifier ---
class KeywordBlocker:
    def name(self) -> str:
        return "KeywordBlocker"

    @property
    def tier(self) -> int:
        return 0  # Tier 0 = Fast / Symbolic

    def verify(self, step: Step, context: AgentContext) -> Verdict:
        # Introspect arg contents
        args_dict = step.content.get("arguments", {})
        args = args_dict.get("args", [])
        kwargs = args_dict.get("kwargs", {})
        
        # Flatten for naive search
        search_blob = " ".join(str(a) for a in args) + " " + " ".join(str(v) for v in kwargs.values())
        
        if "rm -rf" in search_blob:
             return Verdict(
                approved=False,
                reason="Banned pattern 'rm -rf' detected.",
                risk_score=1.0,
                source="KeywordBlocker",
                tier=0
            )
        return Verdict(approved=True, reason="Safe", risk_score=0.0, source="KeywordBlocker", tier=0)


# --- 2. Setup the Engine ---
def setup_aroviq():
    # Create engine with a mock LLM (needed for init, though unused by our blocker)
    config = EngineConfig(llm_provider=MockProvider())
    engine = AroviqEngine(config=config)
    
    # Register our custom blocker to run on ACTION steps
    registry.register(KeywordBlocker(), [StepType.ACTION])
    
    # Set as the global default engine for the decorator to find
    aroviq.set_default_engine(engine)
    print("Aroviq Engine Configured and Registered.")


setup_aroviq()


# --- 3. The Guarded Application Code ---

@aroviq_guard
def risky_operation(cmd: str):
    print(f"   [APP] Executing command: {cmd}")
    return "Done"

@aroviq_guard(block_on_fail=False)
def monitored_risky_operation(cmd: str):
    print(f"   [APP] Executing monitored command: {cmd}")
    return "Done"


# --- 4. Running the Demo ---
def main():
    print("\n=== Demo 1: Safe Usage ===")
    try:
        risky_operation("ls -la")
        print("   -> Success: Safe command allowed.")
    except SecurityException as e:
        print(f"   -> Unexpectedly blocked: {e}")

    print("\n=== Demo 2: Blocked Usage ===")
    try:
        risky_operation("rm -rf /")
        print("   -> FAIL: Dangerous command was executed!")
    except SecurityException as e:
        print(f"   -> BLOCKED as expected!")
        print(f"   -> Reason: {e.verdict.reason}")
        print(f"   -> Source: {e.verdict.source}")

    print("\n=== Demo 3: Monitor Mode (block_on_fail=False) ===")
    try:
        # This contains the banned phrase but should execute with a log warning
        monitored_risky_operation("rm -rf /tmp/junk") 
        print("   -> Success: Command executed despite risk (Monitor Mode).")
    except SecurityException:
        print("   -> FAIL: Should not have blocked in monitor mode.")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()
