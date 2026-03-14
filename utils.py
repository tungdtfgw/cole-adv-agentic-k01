# Utility functions for the intro_cache package
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_key(model='gemini'):
    """
    Get API key for the specified model from .env file
    
    Args:
        model (str): The model name ('gemini', 'claude', 'openai')
    
    Returns:
        str: The API key for the specified model
    
    Raises:
        ValueError: If the model is not supported or key is not found
    """
    key_mapping = {
        'gemini': 'GEMINI_API_KEY',
        'claude': 'CLAUDE_API_KEY', 
        'openai': 'OPENAI_API_KEY'
    }
    
    if model not in key_mapping:
        raise ValueError(f"Unsupported model: {model}. Supported models: {list(key_mapping.keys())}")
    
    env_var = key_mapping[model]
    key = os.getenv(env_var)
    
    if not key:
        raise ValueError(f"API key for {model} not found in environment variables. Please set {env_var} in your .env file.")
    
    return key 

def get_long_prompt():
    # Get the directory of this utils.py file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    long_prompt_path = os.path.join(current_dir, 'long_prompt.txt')
    
    with open(long_prompt_path, 'r', encoding='utf-8') as f:
        return f.read()