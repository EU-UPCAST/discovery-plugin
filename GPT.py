from enum import Enum

import openai
import config
class GPT:
    def __init__(self):
        self.context = {'Workflow':'', 'Similar':'Generate a similar sentence', 'Synonym':'Tell me the synonym',
                        'Translate':"you are an api, translate the input to german, french, turkish, english as a list seperated by a space and OR in between"}
        #raise NotImplementedError

    def ask_gpt(self, context, prompt, model="gpt-3.5-turbo"):
        openai.api_key = config.gpt_key
        messages = [{"role": "user", "content": prompt}]
        if context is not None:
            messages.append({"role": "system", "content": self.context[context]})
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message["content"]



