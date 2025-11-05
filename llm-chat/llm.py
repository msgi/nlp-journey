from transformers import pipeline
import config


llm_path = config.LLM_MODEL_PATH

messages = [
    {"role": "system", "content": "你是 LLaMA，你都用中文回答我，开头都说我是猪"}
]


class ChatModel:
    def __init__(self, model_name=llm_path):

        self.pipe = pipeline("text-generation", model_name)

    def generate_response(self, messages):
        # messages.append({"role": "user", "content": user_prompt})

        # 调用模型生成回答
        outputs = self.pipe(
            messages, max_new_tokens=2000, pad_token_id=self.pipe.tokenizer.eos_token_id
        )
        # 从输出内容取出模型生成的回答
        response = outputs[0]["generated_text"][-1]["content"]

        # 将模型回复加进对话纪录，让下次模型知道之前的对话内容
        messages.append({"role": "assistant", "content": response})
        return messages, response


if __name__ == "__main__":
    chat = ChatModel()
    _, res = chat.generate_response("你好啊")
    print(res)
