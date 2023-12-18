from bs4 import BeautifulSoup
from openai import OpenAI
import requests
import traceback
from dotenv import load_dotenv
import os

from chat import Chat

load_dotenv()  # load environment variables from .env file
api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv('OPENAI_BASE_URL')


GPT_MODEL = "gpt-3.5-turbe"
client = OpenAI(api_key=api_key,base_url=base_url)

chat = Chat(client,GPT_MODEL)

def get_current_weather(location:str, format:str='celsius'):
    '''
    Get the current weather

    location: The city and state, e.g. San Francisco, CA

    format: The temperature unit to use. Infer this from the users location.
        enum: ["celsius", "fahrenheit"]
    '''
    return "晴 20~25"

def get_location():
    '''
    Get the user's location
    '''
    return "beijing"

def exec_code(code:str, expr:str):
    '''
    execute a python code. BeautifulSoup and requests are imported. return exception message when an exception is encountered

    code: The code to execute

    expr: The value to be returned
    '''
    glo = globals()
    try:
        exec(code,glo)
        return repr(eval(expr,glo))
    except:
        return traceback.format_exc()


# chat.add_tool(get_current_weather)
# chat.add_tool(get_location)
# chat.add_tool(exec_code)

# chat.add(chat.req())

print()
print("Enter to confirm")
print("type ''' to input muti lines")
print()

chat.set_settings("Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.")

hold = False
lines = []
print('\033[32muser: ',end='',flush=True)

try:
    while True:
        line = input()
        if line == "'''":
            hold = not hold
            line = '\n'.join(lines[1:])
            lines = []
        if hold:
            lines.append(line)
            # print('- ',end='',flush=True)
        else:
            print('\033[0m')
            chat.call({'role':'user','content':line})
            print('\033[32muser: ',end='',flush=True)
except KeyboardInterrupt:
    print('\033[0m\nbye')
