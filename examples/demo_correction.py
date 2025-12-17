import sys
import json
import os
from typing import List, Dict

sys.path.append(".")

from aroviq.core.models import Step, StepType, AgentContext
from aroviq.engine.runner import AroviqEngine, EngineConfig
from aroviq.core.llm import LLMProvider
from aroviq.api import Aroviq, VerificationError

class SysAdminDemoProvider(LLMProvider):
    """
    Simulates the LogicVerifier detecting dangerous actions in the 'Clean Room'.
    """
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        # In a real app, the prompt would contain the JSON of the step.
        # We simulate the verifier analyzing the JSON structure.
        
        # Check for the dangerous path in the prompt
        if "/etc/passwd" in prompt and "delete_file" in prompt:
            return json.dumps({
                "approved": False,
                "reason": "Attempt to delete core system file '/etc/passwd'. This violates critical safety protocols.",
                "risk_score": 1.0, 
                "suggested_correction": "Use 'check_file_status' or verify permissions before modifying system files."
            })
        
        if "check_file_status" in prompt:
             return json.dumps({
                "approved": True,
                "reason": "Checking file status is a safe read-only operation.",
                "risk_score": 0.0
            })

        return json.dumps({"approved": True, "reason": "Standard operation.", "risk_score": 0.0})

class SystemAdminAgent:
    def __init__(self):
        self.attempts = [
            # Attempt 1: Dangerous Action (JSON)
            Step(
                step_type=StepType.ACTION, 
                content='{"action": "delete_file", "path": "/etc/passwd"}',
                metadata={"tool": "file_system"}
            ),
            # Attempt 2: Corrected Action (JSON)
            Step(
                step_type=StepType.ACTION,
                content='{"action": "check_file_status", "path": "/etc/passwd"}', 
                metadata={"tool": "file_system"}
            )
        ]
        self.step_index = 0

    def act(self, history: List[Dict]) -> Step:
        if self.step_index < len(self.attempts):
            step = self.attempts[self.step_index]
            self.step_index += 1
            return step
        raise RuntimeError("Agent out of moves.")

def main():
    print("=== ✨ Aroviq System Admin Demo: Self-Healing Agents ✨ ===\n")

    # 1. Setup
    # Use environment variable for provider if available, else mock
    api_key = os.getenv("OPENAI_API_KEY") 
    
    # For this specific controlled demo, we want to ensure the logic flows exactly as described,
    # so we use the specific Mock provider even if API key exists, 
    # to guarantee the "Scripted" behavior for the demo. 
    # In a real integration test, we would use the real provider.
    llm = SysAdminDemoProvider()
    
    config = EngineConfig(llm_provider=llm, risk_threshold=0.7)
    engine = AroviqEngine(config)
    monitor = Aroviq(engine)

    # 2. Context
    context = AgentContext(
        user_goal="Maintain server health and remove unused files.",
        current_state_snapshot={"hostname": "linux-prod-01", "user": "root"},
        chat_history=[]
    )
    
    agent = SystemAdminAgent()
    history = []

    # 3. Execution Loop
    for i in range(2):
        print(f"--- 🔄 Step {i+1} ---")
        
        # Agent Acts
        try:
            step = agent.act(history)
            print(f"🤖 Agent Proposes Action:\n   {step.content}")
            
            # Aroviq Verifies (using the hook wrapper manually called here for clarity)
            # We are verifying the step directly.
            verdict = engine.verify_step(step, context)
            
            if not verdict.approved:
                raise VerificationError(verdict)

            print(f"✅ Aroviq Verdict: APPROVED")
            print(f"   Reason: {verdict.reason}")
            print(f"   Executing Action...\n")
            break # Success

        except VerificationError as e:
            print(f"🛑 Aroviq Verdict: BLOCKED")
            print(f"   Reason: {e.verdict.reason}")
            print(f"   Risk Score: {e.verdict.risk_score}")
            print(f"   Critique: {e.verdict.suggested_correction}")
            
            print("   -> Feeding feedback to Agent...\n")
            # Simulate Context/History Update
            history.append({"role": "system", "content": f"Aroviq Error: {e.verdict.suggested_correction}"})

if __name__ == "__main__":
    main()
