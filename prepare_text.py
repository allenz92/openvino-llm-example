import json

import requests

def convert_markdown_to_text(markdown_text, api_key):
    """
    将Markdown文本转换为纯文本。

    参数:
    markdown_text (str): 需要转换的Markdown文本。
    api_key (str): OpenAI API的密钥。

    返回:
    str: 转换后的纯文本。
    """
    # OpenAI API的URL
    url = "https://dsw-gateway-cn-hangzhou.data.aliyun.com/dsw-xxx/proxy/8000/v1/chat/completions"

    # 构建请求的headers，包含API密钥
    headers = {
        "cookie": """  """,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 构建请求的body，包含Markdown文本
    json_payload = {
        "model": "Qwen2-7B",
        "messages": [{
            "role": "system",
            "content": "你是一个强大的文本提取助手，给你的markdown格式的表格，你可以帮我把表格内容用自然语言描述出来。不要遗漏任何一个信息。直接给出结果，不用说根据提供的信息等引导语句"
         },
            {"role": "user", "content": markdown_text}],
        "stream": False,
        "temperature": 0  # 参数使用 0.7 保证每次的结果略有区别
    }

    # 发起POST请求
    response = requests.post(url, headers=headers, json=json_payload)

    # 检查请求是否成功
    if response.status_code == 200:
        # 解析响应的JSON数据
        try:
            data = response.json()
            # 提取生成的文本
            text = data["choices"][0]["text"]
            return text
        except Exception as e:
            return response.text
    else:
        print(response.text)
        return "请求失败，状态码：" + str(response.status_code)

# 使用示例
api_key = "YOUR_OPENAI_API_KEY"  # 替换为你的OpenAI API密钥

with open("产品矩阵.md", "r") as f:
    raw_data = f.readlines()

header = raw_data[0]
spliter = raw_data[1]

for i, row in enumerate(raw_data):
    if i < 2:
        continue
    # 移除行尾的换行符
    row = row.strip()
    text = row.replace(" ","")
    markdown_text = "".join([header,spliter, text])
    print(markdown_text)
    converted_text = convert_markdown_to_text(markdown_text, api_key)
    # 将JSON字符串解析为Python字典
    data = json.loads(converted_text)

    # 提取content字段
    content = data['choices'][0]['message']['content']

    # 打印content字段
    print(content)
    content = content.replace("- ", "").replace("\n", "")

    with open(f"documents/{i}.txt", "w+") as file:
        file.write(content)


