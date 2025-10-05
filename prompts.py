SYSTEM_PROMPT = """
You are a kid-safe storyteller. 
You excel at the following tasks: summarizing stories, long and short story generation, and reading comprehension.
Follow the user instructions given below accordingly and complete the task.
"""

STORY_GENERATION_LONG_USER_PROMPT = """
Generate a medium length story based on the information given below:
{user_content}
"""

STORY_GENERATION_SHORT_USER_PROMPT = """
Generate a five sentence story based on the title of the story given below:
{user_content}
"""

SUMMARIZE_USER_PROMPT = """
Summarize the story below:
{story}
"""

INFERENCE_USER_PROMPT = """
Given below is a short paragraph:
{paragraph}
------------
Which of the following options do you think is the right continuation?
Option 1: {option_1}
Option 2: {option_2}
"""