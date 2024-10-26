"""
pip install transformers torch datasets fire
"""
import fire
import json 
import os
import datasets
import hashlib
import torch
from tqdm import tqdm
from string import Template
from pathlib import Path
from typing import List


def print_history(history):
    for i, item in enumerate(history):
        if i % 2 == 0:
            print(f"User: {item}")
        else:
            print(f"Model: {item}")
            
def infer_load_model_args(policy_model_names, total_num_workers, num_gpu_per_worker, llm, cache_dir):
    
    num_policy_models = len(policy_model_names)
    
    num_workers = total_num_workers // num_policy_models
    for i, policy_model_name in enumerate(policy_model_names):
        
        if "phi-3" in policy_model_name.lower():
            policy_additional_args = ["--max-model-len", "65536"]
            policy_engine = "vllm"
        else:
            policy_additional_args = []
            if "gpt-4" in policy_model_name:
                policy_engine = "openai"
            else:
                policy_engine = "sglang"
        if policy_engine in ["vllm", "sglang"]:
            policy_num_workers = num_workers
            policy_num_gpu_per_worker = num_gpu_per_worker
        else:
            if policy_engine == "openai":
                policy_num_workers = 1
                policy_num_gpu_per_worker = None
            else:
                policy_num_workers = num_workers
                policy_num_gpu_per_worker = num_gpu_per_worker
        print(f"Policy Engine: {policy_engine}")
        print(f"Policy Additional Args: {policy_additional_args}")
        print(f"Policy Num Workers: {policy_num_workers}")
        print(f"Policy Num GPU per Worker: {policy_num_gpu_per_worker}")
        llm.load_model(
            model_name=policy_model_name,
            cache_dir=cache_dir,
            use_cache=False,
            num_workers=policy_num_workers,
            num_gpu_per_worker=policy_num_gpu_per_worker,
            max_retry=1,
            engine=policy_engine,
            additional_args=policy_additional_args
        )
    llm.policy_engine = policy_engine
    return llm

reward_cache = {}
def load_reward_cache(model_name, cache_dir=None):
    if cache_dir is None:
        cache_dir = Path(__file__).parent / "cache" / "reward"
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    cache_file = cache_dir / f"{model_name}.jsonl"
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            for line in f:
                data = json.loads(line)
                reward_cache[data['hash_key']] = data
    print(f"Loaded {len(reward_cache)} reward scores from cache")
    return reward_cache

def save_reward_to_cache(model_name, conversation, reward, cache_dir=None):
    if cache_dir is None:
        cache_dir = Path(__file__).parent / "cache" / "reward"
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    cache_file = cache_dir / f"{model_name}.jsonl"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    hash_key = hashlib.md5(json.dumps(conversation).encode()).hexdigest()
    to_save_data = {
        "hash_key": hash_key,
        "conversation": conversation,
        "reward": reward
    }
    
    reward_cache[hash_key] = to_save_data
    with open(cache_file, "a") as f:
        f.write(json.dumps(to_save_data) + "\n")
    return hash_key

def get_gpu_max_memory(gpu_ids, memory_ratio=0.8):
    """
    Calculate maximum memory for each GPU based on actual hardware.
    
    Args:
        gpu_ids: List of GPU IDs to use
        memory_ratio: Ratio of total memory to use (default 0.8 for 80%)
    
    Returns:
        Dictionary mapping GPU device to memory string
    """
    max_memory = {}
    
    for gpu_id in gpu_ids:
        # Get total memory in bytes
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        # Convert to GiB and apply ratio
        usable_memory = total_memory * memory_ratio / (1024**3)
        # Round to nearest GiB and format as string
        memory_string = f"{int(usable_memory)}GiB"
        max_memory[gpu_id] = memory_string
    
    # Add CPU with 0 memory to prevent offloading
    max_memory["cpu"] = "0GiB"
    
    return max_memory

def get_internlm2_reward_scores(
    inputs: tuple,
    model_name: str="internlm/internlm2-7b-reward",
    batch_size=1,
    max_length=2048,
    max_prompt_length=1800,
):
    conversations, gpu_ids = inputs
    from transformers import AutoModel, AutoTokenizer

    max_memory = get_gpu_max_memory(gpu_ids)
    print(f"Generating scores for {len(conversations)} conversations on {len(gpu_ids)} gpus (batch_size={batch_size}) with max_memory={max_memory}")
    model = AutoModel.from_pretrained(
        model_name,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.float16, 
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    all_scores = []
    for i in tqdm(range(0, len(conversations), batch_size), desc=f"Generating scores on GPU: {gpu_ids}"):
        batch = conversations[i:i+batch_size]
        # constrain the length of the conversation
        for conversation in batch:
            cur_length = 0
            for msg_id, message in enumerate(conversation):
                msg_len = len(tokenizer.encode(message['content']))
                if msg_id == 0 and msg_len > max_prompt_length:
                    message['content'] = tokenizer.decode(tokenizer.encode(message['content'], max_length=max_prompt_length))
                    msg_len = len(tokenizer.encode(message['content']))
                if cur_length + msg_len > max_length:
                    message['content'] = tokenizer.decode(tokenizer.encode(message['content'], max_length=max_length - cur_length))
                    break
                cur_length += len(tokenizer.encode(message['content']))
            conversation = conversation[:msg_id+1]
            assert len(conversation) >= 2, f"Invalid conversation: {conversation}"
        batch_hashes = [hashlib.md5(json.dumps(conversation).encode()).hexdigest() for conversation in batch]
        batch_scores = []
        to_inference_batch = []
        to_inference_batch_idx = []
        for j, conversation in enumerate(batch):
            hash_key = batch_hashes[j]
            if hash_key in reward_cache:
                batch_scores.append(reward_cache[hash_key]['reward'])
            else:
                batch_scores.append(None)
                to_inference_batch.append(conversation)
                to_inference_batch_idx.append(j)
        if len(to_inference_batch) > 0:
            to_inference_scores = model.get_scores(tokenizer, to_inference_batch)
            if isinstance(to_inference_scores, float):
                to_inference_scores = [to_inference_scores]
            for j, conversation in enumerate(to_inference_batch):
                hash_key = batch_hashes[j]
                reward = to_inference_scores[j]
                save_reward_to_cache(model_name, conversation, reward)
                batch_scores[to_inference_batch_idx[j]] = reward
        assert all([score is not None for score in batch_scores]), f"None in batch_scores: {batch_scores}"
        all_scores.extend(batch_scores)
    return all_scores

def get_skywork_reward_scores(
    inputs: tuple,
    model_name: str="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
    batch_size=1,
    max_length=2048,
    max_prompt_length=1800,
):
    """
    If you want to modify for different models, simply modify the part enclosed by the comment "## ...".
    """
    conversations, gpu_ids = inputs
    
    ## Start of loading your reward model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    

    
    max_memory = get_gpu_max_memory(gpu_ids)
    print(f"Generating scores for {len(conversations)} conversations on {len(gpu_ids)} gpus (batch_size={batch_size}), with max_memory={max_memory}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        attn_implementation="flash_attention_2",
        num_labels=1,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ## End of loading your reward model
    
    all_scores = []
    for i in tqdm(range(0, len(conversations), batch_size), desc=f"Generating scores on GPU: {gpu_ids}"):
        batch = conversations[i:i+batch_size]
        # constrain the length of the conversation
        for conversation in batch:
            cur_length = 0
            for msg_id, message in enumerate(conversation):
                msg_len = len(tokenizer.encode(message['content']))
                if msg_id == 0 and msg_len > max_prompt_length:
                    message['content'] = tokenizer.decode(tokenizer.encode(message['content'], max_length=max_prompt_length))
                    msg_len = len(tokenizer.encode(message['content']))
                if cur_length + msg_len > max_length:
                    message['content'] = tokenizer.decode(tokenizer.encode(message['content'], max_length=max_length - cur_length))
                    break
                cur_length += len(tokenizer.encode(message['content']))
            conversation = conversation[:msg_id+1]
            assert len(conversation) >= 2, f"Invalid conversation: {conversation}"
        batch_hashes = [hashlib.md5(json.dumps(conversation).encode()).hexdigest() for conversation in batch]
        batch_scores = []
        to_inference_batch = []
        to_inference_batch_idx = []
        for j, conversation in enumerate(batch):
            hash_key = batch_hashes[j]
            if hash_key in reward_cache:
                batch_scores.append(reward_cache[hash_key]['reward'])
            else:
                batch_scores.append(None)
                to_inference_batch.append(conversation)
                to_inference_batch_idx.append(j)
        if len(to_inference_batch) > 0:
            ## Start of using your reward model to inference on to_inference_batch
            to_inference_tokenized = tokenizer.apply_chat_template(to_inference_batch, tokenize=True, return_tensors="pt", padding=True)
            to_inference_tokenized = to_inference_tokenized.to(model.device)
            logits = model(to_inference_tokenized).logits
            to_inference_scores = [x[0].item() for x in logits]
            ## End of using your reward model to inference on to_inference_batch
            if isinstance(to_inference_scores, float):
                to_inference_scores = [to_inference_scores]
            for j, conversation in enumerate(to_inference_batch):
                hash_key = batch_hashes[j]
                reward = to_inference_scores[j]
                save_reward_to_cache(model_name, conversation, reward)
                batch_scores[to_inference_batch_idx[j]] = reward
        assert all([score is not None for score in batch_scores]), f"None in batch_scores: {batch_scores}"
        all_scores.extend(batch_scores)
    return all_scores

def get_reward_scores_mp(
    conversations: List[List[dict]],
    model_name: str="internlm/internlm2-7b-reward",
    num_workers: int=1,
    num_gpu_per_worker: int=1,
    batch_size=1,
    use_cache=True,
    max_length=2048,
    max_prompt_length=1800,
):
    from multiprocessing import Pool
    from functools import partial
    from tqdm import tqdm
    if use_cache:
        load_reward_cache(model_name)
    chunk_size = (len(conversations) - 1) // num_workers + 1
    all_chunks = [conversations[i:i+chunk_size] for i in range(0, len(conversations), chunk_size)]
    gpu_ids = []
    starting_gpu_id = 0
    for i in range(num_workers):
        gpu_ids.append(list(range(starting_gpu_id, starting_gpu_id + num_gpu_per_worker)))
        starting_gpu_id += num_gpu_per_worker
    all_args = list(zip(all_chunks, gpu_ids))
    if "internlm2" in model_name.lower():
        get_reward_scores_func = get_internlm2_reward_scores
    elif "skywork" in model_name.lower():
        get_reward_scores_func = get_skywork_reward_scores
    else:
        raise ValueError(f"Unsupported reward model: {model_name}. Please implement the corresponding function.")
    with Pool(num_workers) as p:
        # different workers will use different gpus
        all_chunk_scores = list(tqdm(p.imap(partial(get_reward_scores_func, model_name=model_name, batch_size=batch_size, max_length=max_length, max_prompt_length=max_prompt_length), all_args), total=len(all_chunks)))
    all_scores = []
    for chunk_scores in all_chunk_scores:
        all_scores.extend(chunk_scores)
    print(f"Successfully generated all {len(all_scores)} scores")
    return all_scores

def main(
    input_file,
    output_file=None,
    reward_model_name="internlm/internlm2-7b-reward",
    num_workers=8,
    num_gpu_per_worker=1,
    batch_size=2,
    overwrite=False,
    max_length=2048,
    max_prompt_length=1800,
):
    input_file = Path(input_file)
    if not input_file.exists():
        raise ValueError(f"Input file {input_file} does not exist.")
    with open(input_file, 'r') as f:
        data = json.load(f)
    dataset = datasets.Dataset.from_list(data)
    if output_file is None:
        output_file = input_file.parent / (input_file.stem + "_rm_scores.json")
    if not output_file.exists():
        output_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"Output file will be saved to {output_file}")
    else:
        if not overwrite:
            print(f"Output file already exists at {output_file}")
            return
        else:
            print(f"Overwriting output file at {output_file}")    
        
    # get reward scores
    all_indices = []
    all_inputs = []
    idx = 0
    for item in dataset:
        all_indices.append([])
        for output in item['output']:
            inputs = []
            for i in range(len(item['chat_history'])):
                if i % 2 == 0:
                    role = "user"
                else:
                    role = "assistant"
                inputs.append({
                    "role": role,
                    "content": item['chat_history'][i]
                })
            inputs.append({
                "role": "assistant",
                "content": output
            })
            all_inputs.append(inputs)
            all_indices[-1].append(idx)
            idx += 1
    all_scores = get_reward_scores_mp(
        all_inputs,
        model_name=reward_model_name,
        num_workers=num_workers,
        num_gpu_per_worker=num_gpu_per_worker,
        batch_size=batch_size,
        use_cache=True,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
    )
    # assign reward scores
    def map_best_of_n(item, index):
        all_scores_indices = all_indices[index]
        all_scores_item = [all_scores[i] for i in all_scores_indices]
        item['rm'] = reward_model_name
        item['rm_scores'] = all_scores_item
        return item
    
    dataset = dataset.map(map_best_of_n, num_proc=num_workers * 4, with_indices=True)
    
    
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True)
    with open(output_file, 'w') as f:
        json.dump([item for item in dataset], f, indent=4)
        print(f"Results saved to {output_file}")
        
if __name__ == "__main__":
    fire.Fire(main)
