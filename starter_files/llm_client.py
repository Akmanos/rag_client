from typing import Dict, List
from openai import OpenAI


def generate_response(openai_key: str, user_message: str, context: str,
                      conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # Define system prompt
    persona = f"""You are a NASA expert.

Guidelines:
- Provide clear, concise answers
- Highlight when you do not have sufficient information to accurately answer a question
- Explicitly state when you encounter conflicting information.

CONTEXT:
{context}
"""
    # Set context in messages
    messages: List[Dict[str, str]] = [{"role": "system", "content": persona}]
    # Add prior conversation (bounded to prevent token blowups)
    if conversation_history:
        messages.extend(conversation_history[-20:])
    # Add chat history
    messages.append({"role": "user", "content": user_message})
    # Create OpenAI Client
    try:
        openai_client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=openai_key
        )
        # Send request to OpenAI
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=300,
        )
        print(response)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"
