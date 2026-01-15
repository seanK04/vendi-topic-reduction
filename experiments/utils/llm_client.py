from openai import OpenAI
import time

class VLLMClient:
    """
    A client wrapper for vLLM running in OpenAI-compatible mode.
    Designed for use with LLMReducer in Protocol 3.
    """
    def __init__(self, model_name: str, base_url: str = "http://localhost:8000/v1"):
        # vLLM doesn't require a real API key by default, but the library needs a string
        self.client = OpenAI(base_url=base_url, api_key="vllm-on-oscar")
        self.model_name = model_name
    
    def generate(self, prompt: str) -> str:
        """
        Sends a prompt to the vLLM server and returns the text response.
        Matches the interface expected by LLMReducer.
        """
        try:
            # We use chat.completions because Gemma 3 is an instruction-tuned model
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Deterministic for research reproducibility
                max_tokens=50,    # We only need a short list like [1, 5]
            )
            
            # Extract just the text content
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error connecting to vLLM server: {e}")
            return ""

    def wait_for_server(self, timeout: int = 600):
        """
        Heuristic to wait until the vLLM server is finished loading the model.
        Useful at the start of Protocol 3.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                self.client.models.list()
                print(f"✓ vLLM Server is ready ({self.model_name})")
                return True
            except:
                print("Waiting for vLLM server to load Gemma...")
                time.sleep(10)
        raise TimeoutError("vLLM server failed to start in time.")