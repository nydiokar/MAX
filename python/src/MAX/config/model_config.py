from dataclasses import dataclass, field
from typing import Dict

# Anthropic/Local Constants
MODEL_ID_CLAUDE_3_HAIKU= "anthropic.claude-3-haiku-20240307-v1:0"
MODEL_ID_CLAUDE_3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
MODEL_ID_CLAUDE_3_5_SONNET = "anthropic.claude-3-5-sonnet-20240620-v1:0"
MODEL_ID_LLAMA_3_1_8B = "C:/Users/Cicada38/Orchi/Orch/python/src/MAX/config/llama3.18b-instruct-q8_0.gguf"

# Llama Configuration
@dataclass
class LlamaConfig:
    model_path: str = MODEL_ID_LLAMA_3_1_8B
    n_ctx: int = 2048
    n_threads: int = 4
    verbose: bool = False
    system_prompt: str = """You are a helpful AI assistant. Your responses should be:
    - Clear and concise
    - Accurate and well-reasoned
    - Helpful while maintaining safety
    - Based on the provided context when available"""
    
    prompt_templates: Dict[str, str] = field(default_factory=lambda: {
        "classification": """
        Classify the following input into one of these categories and provide a confidence score (0-1):
        Categories: [question, command, statement, greeting, task, clarification]
        
        Input: {input_text}
        
        Respond in the format:
        category:confidence
        
        Classification:""",
        
        "chat": """
        Context: {context}
        Previous conversation: {history}
        User: {input_text}
        Assistant:""",
        
        "task": """
        Analyze the following task and break it down into steps:
        Task: {input_text}
        
        Steps:"""
    })
    
    # Added optional model-specific parameters
    model_params: Dict[str, any] = field(default_factory=lambda: {
        "rope_freq_base": 10000,
        "rope_freq_scale": 1.0,
        "batch_size": 512
    })