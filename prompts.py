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

STORY_GENERATION_CREATIVE_USER_PROMPT = """
Write a creative story based on the following prompt:
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


def format_user_content(task_type: str, user_content) -> str:
    if task_type == "generate_long":
        return STORY_GENERATION_LONG_USER_PROMPT.format(user_content=user_content)
    
    elif task_type == "generate_short":
        return STORY_GENERATION_SHORT_USER_PROMPT.format(user_content=user_content)
    
    elif task_type == "generate_creative":
        return STORY_GENERATION_CREATIVE_USER_PROMPT.format(user_content=user_content)
    
    elif task_type == "summarize":
        return SUMMARIZE_USER_PROMPT.format(story=user_content)
    
    elif task_type == "inference":
        if not isinstance(user_content, dict):
            raise ValueError("user_content must be dict for inference task")
        return INFERENCE_USER_PROMPT.format(
            paragraph=user_content["paragraph"],
            option_1=user_content["option_1"],
            option_2=user_content["option_2"]
        )
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")