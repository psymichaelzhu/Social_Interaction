import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

# Define the prompt
prompt = """You are an expert in social interaction research. Your task is to list potential dimensions related to "dyadic social interaction" that meet the following criteria:
1. Not specific interaction instances (like "co-worker"), but higher-level concepts (like "relationship").
2. These dimensions should encompass a group of different interaction modes, not single specific behaviors.
3. Dimensions should be as independent as possible, avoiding redundancy.


Please organize your response in a hierarchical structure and return it in JSON format for easy processing. Each dimension should include:
- `dimension`: dimension name
- `description`: brief definition of the dimension
- `examples`: several typical interaction types under this dimension

Example output format:
{
    "dimension": "relationship",
    "description": "Types of social relationships between people.",
    "examples": ["parent-child", "friendship", "co-worker"]
},
    {
        "dimension": "activity",
        "description": "the activity people engage in",
        "examples": ["playing instruments", "talking", "fishing"]
    }"""

# Make API call
response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    temperature=1
)

# Print response
print(response.choices[0].message.content)


#%%
#重复这个过程 总结所有的维度
#1. 类型：polar
#2. 