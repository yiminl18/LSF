import os, sys

# Add the current directory to the path so we can import from models
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from models.gpt_41_mini_azure import gpt_41_mini_azure

#this is the models API. You pass the model (name of the model) and prompt, the API will return the response out 
def model(prompt, model_name = 'gpt_41_mini_azure'):
    if(model_name == 'gpt_41_mini_azure'):
        return gpt_41_mini_azure(prompt)
    return 'input model does not exist'


def test_model():
    """
    Test function to verify the model is working correctly.
    """
    print("Testing model...")
    
    # Test prompt
    test_prompt = ["What is the capital of France?", " Please answer in one word."]
    
    try:
        # Test the model
        result = model(test_prompt)
        print(f"Model response: {result}")
        print("✅ Model test successful!")
        return True
    except Exception as e:
        print(f"❌ Model test failed: {str(e)}")
        return False


if __name__ == "__main__":
    test_model()



