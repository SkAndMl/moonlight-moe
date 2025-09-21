from datasets import load_dataset
import tiktoken, pathlib, json, numpy as np, tqdm

tokenizer = tiktoken.get_encoding("gpt2")
with open("config.json", "r") as f:
    config = json.loads(f.read())

NUM_TOKENS_PER_SHARD = 25_000_000
CTX_LENGTH = config["context_length"]
NUM_SEQUENCES_PER_SHARD = NUM_TOKENS_PER_SHARD // CTX_LENGTH

for split in {"train", "test"}:
    ds = load_dataset("wikitext", "wikitext-103-v1", split=split)
    stories = []
    story_content = ""
    for row in ds["text"]:
        if row.count("=") == 2:
            if len(story_content.strip()) > 0:
                stories.append(story_content)
                story_content = ""
        story_content += row

    if len(story_content.strip()) > 0:
        stories.append(story_content)

    shard_id = 0
    buf = []

    out_dir = pathlib.Path(f"shards/{split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for story in tqdm.tqdm(stories, desc="Creating shards..."):

        ids = tokenizer.encode(story) + [tokenizer.eot_token]
        buf.extend(ids)

        if len(buf) >= NUM_TOKENS_PER_SHARD:
            shard_path = open(out_dir / f"shard_{shard_id:05d}.bin", "wb")
            block = np.asarray(buf[:NUM_SEQUENCES_PER_SHARD * CTX_LENGTH], dtype=np.uint16)
            block.tofile(shard_path)
            shard_path.close()

            del buf[:NUM_SEQUENCES_PER_SHARD * CTX_LENGTH]
            shard_id += 1

    if len(buf) > 0:
        shard_path = open(out_dir / f"shard_{shard_id:05d}.bin", "wb")
        block = np.asarray(buf[:NUM_SEQUENCES_PER_SHARD * CTX_LENGTH], dtype=np.uint16)
        block.tofile(shard_path)
        shard_path.close()
        

    print(f"Finished writing to {shard_id+1} shards")