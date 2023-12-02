import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import load_dataset
from prompts import wolframe_prompt
from data_generation.wolframe import WolframePostprocessing
from data_generation.api_checker import check_apis_available
import json
import time
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='do some continuations')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument("--num_devices", type=int, default=8)
    args = parser.parse_args()
    gpt_tokenizer = AutoTokenizer.from_pretrained("Writer/palmyra-small")
    prompt_tokens = gpt_tokenizer(wolframe_prompt, return_tensors="pt")["input_ids"]
    start_tokens = [
        gpt_tokenizer("[")["input_ids"][0],
        gpt_tokenizer(" [")["input_ids"][0],
    ]
    end_tokens = [
        gpt_tokenizer("]")["input_ids"][0],
        gpt_tokenizer(" ]")["input_ids"][0],
    ]  # TODO: keep second?
    api_handler = WolframePostprocessing(start_tokens, end_tokens)
    model = AutoModelForCausalLM.from_pretrained(
        "Writer/palmyra-small",
        low_cpu_mem_usage=True,
    ).cuda()
    dataset = load_dataset("math_dataset", "arithmetic__add_sub_multiple", split="train", streaming=True)
    iter_data = iter(dataset)
    test = False
    counter = 0
    file_counter = 0
    found_examples = 0
    output_dataset = list()
    start_time = time.process_time()
    num_examples = int(25000.0/float(args.num_devices))
    start_count = -1
    if os.path.isfile(f"wolfram_data_{args.device_id}.json"):
        with open(f"wolfram_data_{args.device_id}.json") as f:
            output_dataset = json.load(f)
            start_count = output_dataset[-1]['file_index']
            for item in output_dataset:
                num_examples -= len(item['Wolframe_output'])
    while found_examples < num_examples:
        data = next(iter_data)
        #print(data)
        data["text"]=data["question"]+ " " +data["answer"]
        if file_counter < start_count:
            file_counter += 1
            continue
        if file_counter % args.num_devices != args.device_id:
            file_counter += 1
            continue
        available = check_apis_available(data, gpt_tokenizer)
        test = available.wolframe
        if test:
            #print("in test")
            data_outputs = api_handler.parse_article(data, model, gpt_tokenizer)
            output_dataset.append(
                {
                    "file_index": file_counter,
                    "text": data["text"],
                    "Wolframe_output": data_outputs
                }
            )
            prev_found = found_examples
            found_examples += len(output_dataset[-1]["Wolframe_output"])
            eta_s = (num_examples - found_examples) * (time.process_time()-start_time) / max(1, found_examples)
            eta_m = eta_s // 60
            eta_h = eta_m // 60
            eta_m = eta_m - (eta_h*60)
            eta_s = eta_s - ((eta_m*60) + (eta_h*60*60))
            print(f"Found: {found_examples}/{num_examples}, ETA: {eta_h}H:{eta_m}M:{eta_s}s")
            if found_examples//10 > prev_found//10:
                with open(f"wolfram_data_{args.device_id}.json", 'w') as f:
                    json.dump(output_dataset, f, indent=2)
            counter += 1
        file_counter += 1
    with open(f"wolfram_data_{args.device_id}.json", 'w') as f:
        json.dump(output_dataset, f, indent=2)