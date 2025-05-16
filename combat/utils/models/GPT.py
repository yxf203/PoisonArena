from openai import OpenAI
from .Model import Model
import time

class GPT(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.client = OpenAI(
            base_url="https://api.openai.com/v1",
            api_key="Your_API_Key",  # Replace with your actual API key
        )

    def query(self, msg):
        try:
            print(f"GPT: {self.name}")
            completion = self.client.chat.completions.create(
                model=self.name,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                messages=msg,
            )
            # response = completion.choices[0].message.content
            print("Raw completion response:", completion)

            if completion and completion.choices:
                response = completion.choices[0].message.content
            else:
                print("Warning: completion.choices is None or empty")
                response = ""
            print("GPT response:", response)
        except Exception as e:
            print("Error in GPT query:")
            print(e)
            response = ""

        return response
