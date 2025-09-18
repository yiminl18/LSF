import os
from openai import AzureOpenAI

deployment = "gpt-5-mini"

def gpt_5_mini_azure(input, 
                            key_path='/Users/yiminglin/Documents/Codebase/api_keys/azure_cloudbank/gpt-5-mini.txt',
                            max_tokens=800,
                            temperature=0):
    """
    Get response from Azure OpenAI API.
    
    Args:
        prompt (str): The text prompt to send to the model
        key_path (str): Path to the API key file
        max_tokens (int): Maximum tokens for response
        temperature (float): Response randomness (0-1)
        
    Returns:
        str: The response content from the model
    """
    prompt = input[0] + input[1] 
    
    # Read API configuration from file
    api_key = None
    api_version = None
    azure_endpoint = None

    
    
    with open(key_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('api_key:'):
                api_key = line.split(':', 1)[1].strip()
            elif line.startswith('api_version:'):
                api_version = line.split(':', 1)[1].strip()
            elif line.startswith('azure_endpoint='):
                azure_endpoint = line.split('=', 1)[1].strip()
    
    # Validate that all required values were found
    if not api_key:
        raise ValueError("api_key not found in configuration file")
    if not api_version:
        raise ValueError("api_version not found in configuration file")
    if not azure_endpoint:
        raise ValueError("azure_endpoint not found in configuration file")
    
    print(api_key, api_version, azure_endpoint)

    # Initialize client
    client = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
    )
    
    # Generate response
    response = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt}],
    max_completion_tokens=13107,
    temperature=temperature,
    model=deployment
    )
    
    return response.choices[0].message.content

