import asyncio
from MAX.agents.ollama_agent import OllamaAgent, OllamaAgentOptions
from MAX.utils import Logger

async def test_intent_classification():
    logger = Logger()
    logger.log_header("INTENT CLASSIFICATION TEST")
    
    # Initialize agent
    config = OllamaAgentOptions(
        name="Intent Classifier",
        description="Testing intent classification capabilities",
        model_id="llama3.1:8b-instruct-q8_0"
    )
    
    agent = OllamaAgent(config)
    
    # Test cases
    test_inputs = [
        "What is the capital of France?",
        "Please create a Python script to sort a list.",
        "The weather is nice today.",
        "Can you explain how neural networks work?",
        "Analyze this dataset and provide insights."
    ]
    
    logger.info("\nRunning classification tests:")
    for input_text in test_inputs:
        try:
            intent, confidence = await agent.classify_intent(input_text)
            logger.info(f"\nInput: {input_text}")
            logger.info(f"Classified Intent: {intent}")
            logger.info(f"Confidence: {confidence:.2f}")
        except Exception as e:
            logger.error(f"Error processing '{input_text}': {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_intent_classification())