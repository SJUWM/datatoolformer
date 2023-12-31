import json
from datasets import Dataset
from transformers import AutoTokenizer
import tqdm


if __name__ == '__main__':
    with open("combined_data.json") as f:
        data = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained("Writer/palmyra-small")
    hf_training_data = {"text": []}
    for key in tqdm.tqdm(list(data.keys())):
        sorted_keys = sorted(data[key]["outputs"], key=lambda x:x[0])
        print(sorted_keys)
        tokens = tokenizer(data[key]["text"])["input_ids"]
        output_text = ""
        start = 0
        if len(sorted_keys) < 1:
            continue
        for i in range(len(sorted_keys)):
            #print("Inside for")
            if sorted_keys[i][0] != 0:
                output_text += tokenizer.decode(tokens[start:sorted_keys[i][0]])
                start = sorted_keys[i][0]
            output_text += "<TOOLFORMER_API_START>" + sorted_keys[i][1] + "<TOOLFORMER_API_RESPONSE>" + str(sorted_keys[i][2]) + "<TOOLFORMER_API_END>"
           # print(output_text)
        if start < len(tokens)-1:
            output_text += tokenizer.decode(tokens[start:])
        hf_training_data["text"].append(output_text)
        #print(output_text)
    dataset = Dataset.from_dict(hf_training_data)
    dataset.push_to_hub("SJ1999/datatransformer1",token="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")

