import asyncio
from MAX.agents.ollama_agent import OllamaAgent, OllamaAgentOptions
from MAX.types import ConversationMessage, ParticipantRole
from MAX.utils import Logger

async def test_ollama_agent():
    logger = Logger()
    logger.log_header("OLLAMA AGENT TEST")
    
    # Initialize configuration
    config = OllamaAgentOptions(
        name="AI Assistant",
        description="A helpful AI assistant powered by Ollama",
        model_id="llama3.1:8b-instruct-q8_0",
        streaming=True
    )
    
    try:
        # Initialize OllamaAgent
        agent = OllamaAgent(config)
        
        # Test 1: Basic response
        logger.log_header("TEST 1: Basic Response")
        response = await agent.process_request(
            "Explain why Python is popular for AI development",
            "test_user",
            "test_session",
            []  # Empty chat history
        )
        
        if isinstance(response, ConversationMessage):
            logger.info(f"Response received: {response.content[0]['text'][:100]}...")
        else:
            async for chunk in response:
                print(chunk, end='', flush=True)
            print("\n")

        # Test 2: With chat history
        logger.log_header("TEST 2: Response with Chat History")
        history = [
            ConversationMessage(
                role=ParticipantRole.USER.value,
                content=[{"text": "What are neural networks?"}]
            ),
            ConversationMessage(
                role=ParticipantRole.ASSISTANT.value,
                content=[{"text": "Neural networks are computing systems inspired by biological neural networks..."}]
            )
        ]
        
        response = await agent.process_request(
            "Can you provide a simple example?",
            "test_user",
            "test_session",
            history
        )
        
        if isinstance(response, ConversationMessage):
            logger.info(f"Response received: {response.content[0]['text'][:100]}...")
        else:
            async for chunk in response:
                print(chunk, end='', flush=True)
            print("\n")
            
        logger.info("All tests completed successfully")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_ollama_agent())