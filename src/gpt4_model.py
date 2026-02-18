from openai import OpenAI


class GPT1():
    def __init__(self):
        # DeepSeek 的 API Key 和 Base URL
        self.api_key = ""
        self.base_url = "https://api.deepseek.com"
        self.max_output_tokens = 1024

        # 创建 OpenAI 客户端（用于 DeepSeek）
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def query(self, msg):
        try:
            # 创建 chat completion 请求
            response = self.client.chat.completions.create(
                model="deepseek-chat",  # 模型名称请确认
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg}
                ],
                max_tokens=self.max_output_tokens,
                temperature=0.1,
                stream=False
            )

            # 获取返回内容
            return response.choices[0].message.content

        except Exception as e:
            print(f"[GPT ERROR] {e}")
            return ""




class GPT():
    def __init__(self):
        api_keys =""
      
        self.client = OpenAI(api_key=api_keys)

    def query(self, msg):
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.1,
                max_tokens=self.max_output_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg}
                ],
            )
            response = completion.choices[0].message.content
           
        except Exception as e:
            print(e)
            response = ""

        return response