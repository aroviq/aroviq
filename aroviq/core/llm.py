from abc import ABC, abstractmethod
import os
from typing import Optional

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generates a response from the LLM."""
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        # importing inside to avoid hard dependency if not used
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self._has_client = True
        except ImportError:
            self.client = None
            self._has_client = False

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        if not self._has_client:
            raise ImportError("OpenAI python package is not installed. Please run `pip install openai`.")
        
        if not self.api_key:
             raise ValueError("OpenAI API key is missing. Please set OPENAI_API_KEY environment variable.")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            # Fallback or re-raise
            raise RuntimeError(f"OpenAI API call failed: {e}")

class MockProvider(LLMProvider):
    """A mock provider for testing without API keys."""
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        # Return a valid JSON verdict for testing purposes
        return '{"approved": true, "reason": "Mock approval from Clean Room.", "risk_score": 0.0}'
