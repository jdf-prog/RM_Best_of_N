"""
pip install llm-engines datasets fire
"""
import fire
import json 
import os
import datasets
import hashlib
import torch
from eval_utils import EVAL_FUNC_MAP, extract_answer_from_prediction
from tqdm import tqdm
from llm_engines import LLMEngine
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
            # print(len(to_inference_batch))
            # print(to_inference_batch)
            to_inference_tokenized = tokenizer.apply_chat_template(to_inference_batch, tokenize=True, return_tensors="pt", padding=True)
            # to cuda
            # print("to_inference_tokenized.shape", to_inference_tokenized.shape)
            to_inference_tokenized = to_inference_tokenized.to(model.device)
            logits = model(to_inference_tokenized).logits
            # print(logits)
            # print("logits.shape", logits.shape)
            # exit(1)
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

zeroeval_template = """## Question: 

${question}

## Instruction 

Please answer this question by first reasoning and then providing your answer.
Present your reasoning and solution in the following json format. 
Please show your final answer in the `answer` field, e.g.,`"answer": "42"`.

```json
{
    "reasoning": "___",
    "answer": "___"
}
```"""

def load_dataset(task, input_file=None):
    if not input_file:
        input_file = Path("../ZeroEval") / 'result_dirs' / task / "Meta-Llama-3-8B-Instruct.json"
        if not input_file.exists():
            dataset = datasets.load_dataset("DongfuJiang/zeroeval", task.replace("-", "_"), split='test')
            def map_chat_history(item, index):
                item['session_id'] = index
                item['chat_history'] = [Template(zeroeval_template).substitute(question=item['question'])]
                return item
            dataset = dataset.map(map_chat_history, with_indices=True)
        else:
            with open(input_file, 'r') as f:
                data = json.load(f)
            dataset = datasets.Dataset.from_list(data)
    else:
        input_file = Path(input_file)
        if not input_file.exists():
            raise ValueError(f"Input file {input_file} does not exist.")
        with open(input_file, 'r') as f:
            data = json.load(f)
        dataset = datasets.Dataset.from_list(data)
    return dataset

def main(
    task="gsm",
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    reward_model_name="internlm/internlm2-7b-reward",
    input_file=None,
    cache_dir="./llm_engines_cache",
    num_workers=8,
    num_gpu_per_worker=1,
    k=16,
    results_dir="results",
    overwrite=False,
):
    dataset = load_dataset(task, input_file)
        
    policy_models = model_name.split(",")
    
    if len(policy_models) == 1:
        results_dir = Path(results_dir) / task / policy_models[0].replace("/", "_")
    else:
        name = "ensemble_" + "_".join([policy_model.split("/")[-1][0] for policy_model in policy_models])
        results_dir = Path(results_dir) / task / name 
    results_dir = results_dir / "rm_best_of_n" / reward_model_name.replace("/", "_") 
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    output_file = results_dir / f"rm_best_of_{k}.json"
    generation_output_file = results_dir.parent.parent / "sampled_outputs" / f"samples_of_{k}.json"
    
    if not output_file.exists() or overwrite:
        
        if not generation_output_file.exists():
            llm = LLMEngine()
            llm = infer_load_model_args(
                policy_models, num_workers, num_gpu_per_worker, llm, cache_dir
            )
            from eval_utils import judge_llm
            judge_llm.load_model(
                model_name="gpt-4o-mini",
                engine="openai",
                use_cache=True,
                cache_dir=cache_dir,
            )
            
            generation_kwargs = {
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": 2048,
                "timeout": 120,
                "n": k,
            }
            eval_item = EVAL_FUNC_MAP[task]
            
            def map_generate(item, index):
                policy_model = policy_models[index % len(policy_models)]
                problem = item['chat_history'][0]
                reference_answer = item['answer'] if 'answer' in item else item['correct_answer']
                policy_model_answers = llm.call_model(policy_model, messages=problem, **generation_kwargs)
                policy_model_answers = [x or str(x) for x in policy_model_answers]
                
                to_eval_item = item.copy()
                to_eval_item['is_timeout'] = False
                to_eval_item['generator'] = f"{policy_model}"
                to_eval_item['configs'] = generation_kwargs
                to_eval_item['configs']['engine'] = llm.policy_engine
                
                all_correct = []
                all_reasons = []
                all_extracted_answers = []
                for raw_answer in policy_model_answers:
                    to_eval_item['output'] = [raw_answer]
                    correct, reason = eval_item(to_eval_item)
                    extracted_answer = extract_answer_from_prediction(raw_answer, policy_model)
                    all_correct.append(correct)
                    all_reasons.append(reason)
                    all_extracted_answers.append(str(extracted_answer))
                to_eval_item['all_correct'] = all_correct
                to_eval_item['all_extracted_answers'] = all_extracted_answers
                to_eval_item['all_outputs'] = policy_model_answers
                to_eval_item['reason'] = all_reasons
                return to_eval_item
        
            dataset = dataset.map(map_generate, num_proc=num_workers * 4, with_indices=True, desc=f"Sampling {k} from {policy_models}")
            llm.unload_model()
            
            greedy_kwargs = {
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 2048,
                "timeout": 120,
            }
            llm = infer_load_model_args(
                policy_models, num_workers, num_gpu_per_worker, llm, cache_dir, use_cache=True
            )
            def map_greedy(item, index):
                policy_model = policy_models[index % len(policy_models)]
                problem = item['chat_history'][0]
                reference_answer = item['answer'] if 'answer' in item else item['correct_answer']
                greedy_answer = llm.call_model(policy_model, messages=problem, **greedy_kwargs)
                to_eval_item = item
                to_eval_item['output'] = [greedy_answer]
                correct, reason = eval_item(to_eval_item)
                to_eval_item['greedy_correct'] = correct
                to_eval_item['greedy_output'] = greedy_answer
                return to_eval_item
            dataset = dataset.map(map_greedy, num_proc=num_workers * 4, with_indices=True, desc=f"Greedy Sampling")
            
            generation_output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(generation_output_file, 'w') as f:
                json.dump([item for item in dataset], f, indent=4)
                print(f"Samples saved to {generation_output_file}")
            for policy_model in policy_models:
                llm.unload_model(policy_model)
        else:
            print(f"Samples already exist at {generation_output_file}")
            with open(generation_output_file, 'r') as f:
                data = json.load(f)
            dataset = datasets.Dataset.from_list(data)
        # get reward scores
        all_indices = []
        all_inputs = []
        idx = 0
        for item in dataset:
            all_indices.append([])
            for output in item['all_outputs']:
                all_inputs.append([
                    {
                        "role": "user",
                        "content": item['chat_history'][0]
                    },
                    {
                        "role": "assistant",
                        "content": output
                    }
                ])
                all_indices[-1].append(idx)
                idx += 1
        all_scores = get_reward_scores_mp(
            all_inputs,
            model_name=reward_model_name,
            num_workers=num_workers,
            num_gpu_per_worker=num_gpu_per_worker,
            batch_size=2,
            use_cache=True,
            max_length=2048,
            max_prompt_length=1800,
        )
        # assign reward scores
        def map_best_of_n(item, index):
            all_scores_indices = all_indices[index]
            all_scores_item = [all_scores[i] for i in all_scores_indices]
            best_score_index = all_scores_item.index(max(all_scores_item))
            item['correct'] = item['all_correct'][best_score_index]
            item['reason'] = item['reason'][best_score_index]
            item['output'] = [item['all_outputs'][best_score_index]]
            item['rm'] = reward_model_name
            item['rm_scores'] = all_scores_item
            return item
        
        dataset = dataset.map(map_best_of_n, num_proc=num_workers * 4, with_indices=True)
        
        
        if not output_file.parent.exists():
            output_file.parent.mkdir(parents=True)
        with open(output_file, 'w') as f:
            json.dump([item for item in dataset], f, indent=4)
            print(f"Results saved to {output_file}")
    else:
        print(f"Results already exist at {output_file}")
        with open(output_file, 'r') as f:
            data = json.load(f)
        dataset = datasets.Dataset.from_list(data)
        
    
    # count success rate
    # first try
    success_output_file = results_dir / f"rm_best_of_{k}.txt"
    with open(success_output_file, 'w') as f:
        
        print("---" * 20, file=f)
        num_success = sum([item['all_correct'][0] for item in dataset])
        num_total = len(dataset)
        print(f"First Sample Success Rate: {num_success} / {num_total} ({num_success/num_total*100:.2f}%)", file=f)
        
        # random sample
        import random
        random.seed(42)
        print("---" * 20, file=f)
        num_success = sum([random.choice(item['all_correct']) for item in dataset])
        num_total = len(dataset)
        print(f"Random Sample Success Rate: {num_success} / {num_total} ({num_success/num_total*100:.2f}%)", file=f)
        
        # greedy
        print("---" * 20, file=f)
        num_success = sum([item['greedy_correct'] for item in dataset])
        num_total = len(dataset)
        print(f"Greedy Success Rate: {num_success} / {num_total} ({num_success/num_total*100:.2f}%)", file=f)
        
        # final best of n
        print("---" * 20, file=f)
        num_success = sum([item['correct'] for item in dataset])
        num_total = len(dataset)
        print(f"Final Best of {k} Success Rate: {num_success} / {num_total} ({num_success/num_total*100:.2f}%)", file=f)
        
        
        
    print(f"Results saved to {success_output_file}")
if __name__ == "__main__":
    fire.Fire(main)
