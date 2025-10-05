import datasets, re, random, tiktoken, util, prompts, os, json
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

SAMPLE_SIZE_TARGET_FOR_LONG_STORY_GENERATION = 9440 # 60% of 16400
SAMPLE_SIZE_TARGET_FOR_SUMMARY = 1640 # 10% of 16400
SAMPLE_SIZE_TARGET_FOR_SHORT_STORY_GENERATION = 1640 # 10% of 16400
SAMPLE_SIZE_TARGET_FOR_INFERENCE = 3278 # 20% of 16400

tokenizer = tiktoken.get_encoding("gpt2")


def parse_sections(text, separate_section):
    section_pattern = r'(\w+):\s*(.*?)(?=\n\w+:|$)'
    sections = re.findall(section_pattern, text, re.DOTALL)
    
    story_section = ""
    other_sections = []
    
    for section_name, content in sections:
        if section_name == separate_section:
            story_section = f"{section_name}: {content.strip()}"
        else:
            other_sections.append(f"{section_name}: {content.strip()}")
    
    return other_sections, story_section


def get_sft_data_from_tinystories() -> dict:
    ds = datasets.load_dataset("skeskinen/TinyStories-Instruct-hf", token=os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    idxs = set(range(len(ds["train"])))
    story_generation_count, summary_count = 0, 0
    sft_dict = {
        "user_content": [],
        "task": [],
        "assistant_content": []
    }
    while not (story_generation_count == SAMPLE_SIZE_TARGET_FOR_LONG_STORY_GENERATION and summary_count == SAMPLE_SIZE_TARGET_FOR_SUMMARY):
        random_idx = random.choice(list(idxs))
        other_sections, story_section = parse_sections(ds["train"][random_idx]["text"], "Story")
        if len(other_sections) == 1 and "Summary:" in other_sections[0] and summary_count < SAMPLE_SIZE_TARGET_FOR_SUMMARY:
            user_content = prompts.SUMMARIZE_USER_PROMPT.format(story=story_section)
            tokens = tokenizer.encode(
                util.format_data_for_sft(prompts.SYSTEM_PROMPT, user_content, other_sections[0]),
                allowed_special="all"
            )
            if len(tokens) > 513:
                continue
            sft_dict["user_content"].append(story_section)
            sft_dict["task"].append("summarize")
            sft_dict["assistant_content"].append(other_sections[0])
            summary_count += 1
        elif story_generation_count < SAMPLE_SIZE_TARGET_FOR_LONG_STORY_GENERATION:
            other_sections = [_ for _ in other_sections if "Summary" not in _]
            user_content = prompts.STORY_GENERATION_LONG_USER_PROMPT.format(user_content="\n".join(other_sections))
            tokens = tokenizer.encode(
                util.format_data_for_sft(prompts.SYSTEM_PROMPT, user_content, story_section),
                allowed_special="all"
            )
            if len(tokens) > 513:
                continue
            sft_dict["user_content"].append("\n".join(other_sections))
            sft_dict["task"].append("generate_long")
            sft_dict["assistant_content"].append(story_section)
            story_generation_count += 1

        idxs.remove(random_idx)
        if len(idxs) == 0:
            break

    return sft_dict


def get_sft_data_from_roc():
    ds = datasets.load_dataset("igormorgado/ROCStories2018", token=os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    idxs = set(range(len(ds["train"])))
    sft_dict = {
        "user_content": [],
        "task": [],
        "assistant_content": []
    }
    while len(sft_dict["user_content"]) <= SAMPLE_SIZE_TARGET_FOR_SHORT_STORY_GENERATION:
        random_idx = random.choice(list(idxs))
        user_content = prompts.STORY_GENERATION_SHORT_USER_PROMPT.format(user_content=ds["train"][random_idx]["storytitle"])
        assistant_content = "\n".join([ds["train"][random_idx][f"sentence{i}"] for i in range(1, 6)])
        tokens = tokenizer.encode(
            util.format_data_for_sft(prompts.SYSTEM_PROMPT, user_content, assistant_content),
            allowed_special="all"
        )
        if len(tokens) > 513:
            continue
        
        sft_dict["user_content"].append(ds["train"][random_idx]["storytitle"])
        sft_dict["task"].append("generate_short")
        sft_dict["assistant_content"].append(assistant_content)
        idxs.remove(random_idx)

        if len(idxs) == 0:
            break

    return sft_dict


def get_sft_data_from_story_cloze():
    ds = datasets.load_dataset("lecslab/story_cloze", token=os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    ds = datasets.concatenate_datasets([ds["train"], ds["test"]])
    idxs = set(range(len(ds)))
    sft_dict = {
        "user_content": [],
        "task": [],
        "assistant_content": []
    }
    while len(sft_dict["user_content"]) <= SAMPLE_SIZE_TARGET_FOR_INFERENCE:
        random_idx = random.choice(list(idxs))
        prompt, chosen, rejected = ds[random_idx]["prompt"], ds[random_idx]["chosen"], ds[random_idx]["rejected"]
        if random.random() > 0.5:
            user_content = prompts.INFERENCE_USER_PROMPT.format(paragraph=prompt, option_1=rejected, option_2=chosen)
        else:
            user_content = prompts.INFERENCE_USER_PROMPT.format(paragraph=prompt, option_1=chosen, option_2=rejected)
        assistant_content = chosen
        tokens = tokenizer.encode(
            util.format_data_for_sft(prompts.SYSTEM_PROMPT, user_content, assistant_content),
            allowed_special="all"
        )
        if len(tokens) > 513:
            continue
        
        sft_dict["user_content"].append({"paragraph": prompt, "option_1": chosen, "option_2": rejected})
        sft_dict["task"].append("inference")
        sft_dict["assistant_content"].append(assistant_content)
        idxs.remove(random_idx)

        if len(idxs) == 0:
            break
    
    return sft_dict


if __name__ == "__main__":

    sft_dict = get_sft_data_from_tinystories()
    _tmp = get_sft_data_from_roc()
    for key in _tmp:
        sft_dict[key].extend(_tmp[key])
    _tmp = get_sft_data_from_story_cloze()
    for key in _tmp:
        sft_dict[key].extend(_tmp[key])
    
    with open("sft_data.json", "w") as f:
        json.dump(sft_dict, f)