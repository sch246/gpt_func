
import json
from typing import Callable
from typing import Callable, List, Dict
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from tool import Tool


class MessageStream:
    def __init__(self, res:Stream[ChatCompletionChunk]) -> None:
        delta = next(res).choices[0].delta
        self._iter = res
        self.role = delta.role
        self.tool_calls = delta.tool_calls
        self.msg = ChatCompletionMessage(
            role=self.role,
            tool_calls=self.tool_calls,
            content=None if self.tool_calls else ''
            )

    def __iter__(self):
        return self
    def __next__(self):
        delta = next(self._iter).choices[0].delta
        if self.tool_calls:
            if delta.tool_calls is None:
                raise StopIteration
            delta_text = delta.tool_calls[0].function.arguments
            self.msg.tool_calls[0].function.arguments += delta_text
            return delta_text
        else:
            if delta.content is None:
                raise StopIteration
            delta_text = delta.content
            self.msg.content += delta_text
            return delta_text



role_to_color = {
    "system": "red",
    "user": "green",
    "assistant": "blue",
    "tool": "magenta",
}
def show_args(args):
    return ', '.join([f'{k}={repr(v)}' for k, v in args.items()])
def show_tool_calls(tool_calls):
    return ''.join(map(lambda s:f'\n    {s["function"]["name"]}({show_args(json.loads(s["function"]["arguments"]))})', tool_calls))
def pprint(message:dict | ChatCompletionMessage | MessageStream):
    '''
    打印 dict, 普通消息, 或者流式消息, 然后返回
    流式消息会转换为普通消息
    '''
    if isinstance(message, MessageStream):
        role = message.role
        tool_calls = message.tool_calls
        if message.tool_calls:
            print(colored(f"assistant called: {tool_calls[0].function.name} ", "yellow"),end='', flush=True)
            for delta in message:
                print(colored(delta, "yellow"),end='', flush=True)
        else:
            print(colored(f"assistant: ", role_to_color[role]),end='', flush=True)
            for delta in message:
                print(colored(delta, role_to_color[role]),end='', flush=True)
        print('\n')
        return message.msg
    else:
        if (isinstance(message,ChatCompletionMessage)):
            msg = message.dict()
        else:
            msg = message
        role = msg.get('role')
        tool_calls = msg.get('tool_calls')
        content = msg.get('content')
        name = msg.get('name')
        if role == "system":
            print(colored(f"system: {content}\n", role_to_color[role]))
        elif role == "user":
            print(colored(f"user: {content}\n", role_to_color[role]))
        elif role == "assistant" and tool_calls:
            print(colored(f"assistant called: {show_tool_calls(tool_calls)}\n", "yellow"))
        elif role == "assistant" and not tool_calls:
            print(colored(f"assistant: {content}\n", role_to_color[role]))
        elif role == "tool":
            print(colored(f"function ({name}): {content}\n", role_to_color[role]))
        else:
            print('else:',msg)
        return message



class Chat:
    def __init__(self, client, settings=None, model: str = "gpt-4-1106-preview") -> None:
        self.client: OpenAI = client
        self.model: str = model
        self.tools: Dict[str, Tool] = {}
        self.messages: List[dict] = []
        self.settings = [{'role':'system','content':settings}] if settings is not None else []

    def set_settings(self, settings:str):
        """
        set settings for AI

        :param settings: The setting string described in natural language.
        :return: self
        """
        self.settings = [{'role':'system','content':settings}]
        return self

    def add_tool(self, call: Callable, name: str = None) -> None:
        """
        Add a callable tool to the chat system.

        :param call: The callable tool to be added.
        :param name: The name of the tool. Defaults to the function name if not provided.

        加入的函数必须有符合条件的注释，并且做好类型标记，可选参数会被检测出来

        支持的类型请查看 Tool.types

        它本质是 set tool 而不是 add tool ，同样名字的却不同功能的函数请设定不同的 name

        例:

        def get_current_weather(location:str, format:str='celsius'):
            '''
            Get the current weather

            location: The city and state, e.g. San Francisco, CA

            format: The temperature unit to use. Infer this from the users location.
                enum: ["celsius", "fahrenheit"]
            '''
        """
        name = name if name is not None else call.__name__
        self.tools[name] = Tool(call, name)

    def add(self, msg: dict | MessageStream) -> dict | ChatCompletionMessage:
        """
        Print, transform and add a message to the chat history, then return the transformed message
        流式消息会被转换为普通消息

        :param msg: The message to be added.
        :return: The added message.
        """
        msg = pprint(msg)
        self.messages.append(msg)
        return msg

    def call(self, message: dict, tool_choice: str = "auto", model: str = None) -> ChatCompletionMessage:
        """
        Process a message, make tool calls if necessary, and return the response message.

        :param message: The message to process.
        :param tool_choice: The tool choice mode.
        :param model: The model to use, defaults to the class model.
        :return: The response message.
        """
        tools = [v.description for v in self.tools.values()]
        model = model if model is not None else self.model
        self.add(message)
        res_msg = self.add(self.req(tools, tool_choice, model))
        tool_calls = res_msg.tool_calls
        while tool_calls:
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                self.add({
                    "role": "tool",
                    "name": function_name,
                    "content": self.tools[function_name].call(**json.loads(tool_call.function.arguments)),
                    "tool_call_id": tool_call.id,
                })
            res_msg = self.add(self.req(tools, tool_choice, model))
            tool_calls = res_msg.tool_calls
        return res_msg

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def req(self, tools: List[str] = None, tool_choice: str = 'auto', model: str = None) -> MessageStream:
        """
        Make a request to the chat model with streaming.

        :param tools: The list of tools available.
        :param tool_choice: The tool choice mode.
        :param model: The model to use, defaults to the class model.
        :return: The chat Stream.
        """
        messages = self.settings + self.messages
        tools = tools if tools is not None else [v.description for v in self.tools.values()]
        model = model if model is not None else self.model
        if tools:
            return MessageStream(self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                stream=True  # 开启流式响应
            ))
        else:
            return MessageStream(self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True
            ))


if __name__=='__main__':
    import os
    from dotenv import load_dotenv
    load_dotenv()  # load environment variables from .env file
    api_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('OPENAI_BASE_URL')

    GPT_MODEL = "gpt-4-1106-preview"
    client = OpenAI(api_key=api_key,base_url=base_url)
    chat = Chat(client,GPT_MODEL)
    chat.add({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
    chat.add({'role':'user','content':'awa'})
    print(chat.add(chat.req()))
    print(chat.call({'role':'user','content':'nothing.'}))
