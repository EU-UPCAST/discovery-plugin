from enum import Enum

import openai

class GPT:
    def __init__(self, key):
        self.key = key
        self.context = {'Workflow':'', 'Similar':'Generate a similar sentence', 'Synonym':'Tell me the synonym'}
        #raise NotImplementedError

    def ask_gpt(self, context, prompt, model="gpt-3.5-turbo"):
        openai.api_key = self.key
        messages = [{"role": "user", "content": prompt}]
        if context is not None:
            messages.append({"role": "system", "content": self.context[context]})
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message["content"]
