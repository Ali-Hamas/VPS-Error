"""
LLM Chat stub implementation using OpenAI directly.
This replaces the emergentintegrations LLM chat functionality.
"""

from dataclasses import dataclass
from typing import Optional
import openai


@dataclass
class UserMessage:
    """User message for chat"""
    text: str


class LlmChat:
    """LLM Chat implementation using OpenAI"""
    
    def __init__(self, api_key: str, session_id: str, system_message: str = ""):
        self.api_key = api_key
        self.session_id = session_id
        self.system_message = system_message
        self.model = "gpt-4o-mini"  # Default model
        self.provider = "openai"
    
    def with_model(self, provider: str, model: str) -> "LlmChat":
        """Set the model to use"""
        self.provider = provider
        # Map old model names to current ones
        model_mapping = {
            "gpt-5-mini": "gpt-4o-mini",
            "gpt-5": "gpt-4o",
            "gpt-4-turbo": "gpt-4-turbo",
        }
        self.model = model_mapping.get(model, model)
        return self
    
    async def send_message(self, message: UserMessage) -> str:
        """Send a message and get response"""
        try:
            client = openai.AsyncOpenAI(api_key=self.api_key)
            
            messages = []
            if self.system_message:
                messages.append({"role": "system", "content": self.system_message})
            messages.append({"role": "user", "content": message.text})
            
            response = await client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"AI service temporarily unavailable: {str(e)}"
