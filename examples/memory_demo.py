import asyncio
from MAX.agents import RecursiveThinkerAgent, RecursiveThinkerOptions
from MAX.config.models import AnthropicModels

async def demo_conversation_memory():
    """Demonstrate the agent's memory capabilities"""
    
    # Initialize agent
    options = RecursiveThinkerOptions(
        name="RecursiveThinker",
        description="A recursive thinking agent",
        model_id=AnthropicModels.HAIKU,
        streaming=False
    )
    agent = RecursiveThinkerAgent(options)
    
    # Example conversation flow
    conversations = [
        "My name is Alice and I love pizza",
        "What's my name?",
        "What food do I like?",
        "Tell me about myself"
    ]
    
    chat_history = []
    for message in conversations:
        print(f"\nUser: {message}")
        response = await agent.process_request(
            input_text=message,
            user_id="demo_user",
            session_id="demo_session",
            chat_history=chat_history
        )
        print(f"Assistant: {response.content[0]['text']}")
        chat_history.append(response)

if __name__ == "__main__":
    asyncio.run(demo_conversation_memory()) 