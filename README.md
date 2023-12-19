# gpt_func
调用gpt函数的一些函数，测试用

测试用例

```python
chat = Chat()
chat.set_settings(["Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."])

def get_current_weather(location:str, format:str='celsius'):
    '''
    Get the current weather

    location: The city and state, e.g. San Francisco, CA

    format: The temperature unit to use. Infer this from the users location.
        enum: ["celsius", "fahrenheit"]
    '''
    return "晴 20~25"

chat.add_tool(get_current_weather)

# 加进聊天上下文
chat.add({'role':'user','content':'今天天气怎么样啊'})
# 调用
res1 = chat.add(chat.req())
# 一次调用
res2 = chat.call({'role':'user','content':'北京的天气'})
# 返回的是 ChatCompletionMessage
print(res1.content, res2.content)
```

