# https://www.gradio.app/guides/creating-a-chatbot-fast
import gradio as gr
import time


text = """
    response = 'from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
import openai
import gradio as gr

os.environ["OPENAI_API_KEY"] = "sk-..."  # Replace with your key

llm = ChatOpenAI(temperature=1.0, model='gpt-3.5-turbo-0613')

def predict(message, history):
    history_langchain_format = []
    for msg in history:
        if msg['role'] == "user":
            history_langchain_format.append(HumanMessage(content=msg['content']))
        elif msg['role'] == "assistant":
            history_langchain_format.append(AIMessage(content=msg['content']))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = llm(history_langchain_format)
    return gpt_response.content

gr.ChatInterface(predict, type="messages").launch()'
"""

def predict(message, history):
    print(f"predict for message {message}")
    buf = ''
    for c in text:
        buf += c
        yield buf
        time.sleep(0.1)

gr.ChatInterface(predict, type="messages").launch()
