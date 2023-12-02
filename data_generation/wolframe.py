import torch
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedModel
)
from tools import WolframAlphaCalculator
from prompts import wolframe_prompt
from typing import List
from data_generation.base_api import APICallPostprocessing

N = 32  # SEQ Len
M = 16  # Min Loss Span To Consider
MAX_LEN = 1024  # Maximum Wolframe length

class WolframePostprocessing(APICallPostprocessing):
    def __init__(
    self,
    start_tokens: List[int],
    end_tokens: List[int],
    minimum_percentage: float = 0.0,
    ):
        self.wolframe = WolframAlphaCalculator
        self.api_text = "Wolframe("
        super().__init__(start_tokens, end_tokens, minimum_percentage)

    def add_api_calls(
        self,
        candidate: int,
        outputs: dict,
        texts_to_test: List[str],
        tokenizer: PreTrainedTokenizerBase,
        input_tokens: torch.Tensor,
        input_start: int,
        nums_to_keep: List[int],
        base_loss: float,
        *args,
        **kwargs
    ):
        generated_texts = list()
        max_token_len = N
        max_token_len_base = N
        print("\n**in add api_calls**")
        #print("Texts to test: ", texts_to_test)
        # print("Outputs dict: ")
        # print(outputs)
        # print("*****")
        for j in range(len(outputs)):
            outputs[j]["Wolframe"] = outputs[j]["generated_text"].replace(
                texts_to_test[candidate], ""
            )
            print("after adding Walframe key outputs[j]: ")
            #print("Outputs dict: ")
            print(outputs[j])
            print("*****\n")
            outputs[j]["Generated"] = outputs[j]["generated_text"].split("Output:")[-1]
            print("Generated: ",outputs[j]["Generated"] )
            if "]" in outputs[j]["Wolframe"]:
                outputs[j]["Wolframe"] = (
                    outputs[j]["Wolframe"].replace("Wolframe(", "").split("]")[0]
                )
                if ")" in outputs[j]["Wolframe"]:
                    outputs[j]["Wolframe"] = outputs[j]["Wolframe"].split(")")[0]
                outputs[j]["Wolframe_text"] = (
                    "[Wolframe(" + outputs[j]["Wolframe"] + ")"
                )
                base_inputs = tokenizer(
                    outputs[j]["Wolframe_text"] + "]" + "\n",
                    return_tensors="pt",
                )["input_ids"].cuda()
                print("Final query sent to api:", outputs[j]["Wolframe"])
                try:
                    outputs[j]["Wolframe"] = self.wolframe(outputs[j]["Wolframe"])
                except (ValueError, TypeError, ZeroDivisionError):
                    continue
                if outputs[j]["Wolframe"] is None:
                    continue
                outputs[j]["Wolframe_output"] = [outputs[j]["Wolframe_text"][1:], str(outputs[j]["Wolframe"])]
                print("Wolframe_output")
                print(outputs[j]["Wolframe_output"])
                outputs[j]["Wolframe_text"] = (
                    outputs[j]["Wolframe_text"]
                    + "->"
                    + str(outputs[j]["Wolframe"])
                    + "]"
                )
                test_inputs = tokenizer(
                    outputs[j]["Wolframe_text"] + "\n",
                    return_tensors="pt",
                )["input_ids"].cuda()
                print("Wolframe_text after all modifications:",  outputs[j]["Wolframe_text"])
                test_inputs = torch.concat(
                    [
                        test_inputs.cuda(),
                        input_tokens[:, input_start:].cuda(),
                    ],
                    dim=1,
                )
                if test_inputs.shape[1] > MAX_LEN:
                    continue
                base_inputs = torch.concat(
                    [
                        base_inputs.cuda(),
                        input_tokens[:, input_start:].cuda(),
                    ],
                    dim=1,
                )
                max_token_len = max(max_token_len, test_inputs.shape[1])
                max_token_len_base = max(max_token_len_base, test_inputs.shape[1])
                generated_texts.append(
                    [
                        test_inputs,
                        base_inputs,
                        nums_to_keep[candidate],
                        base_loss,
                        outputs[j],
                    ]
                )
        return generated_texts, max_token_len, max_token_len_base
    
    def parse_article(
        self, data: dict, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase
    ):
        print("***In Parse Article***")
        outputs = list()
        tokens = tokenizer(data["text"], return_tensors="pt")["input_ids"]
        global N
        N= tokens.shape[1] - 1
        global M
        if N < M:
            M = N-2
        for i in range((tokens.shape[1]-1)//N):
            if (N * (i + 1)) > tokens.shape[1]:
                continue
            input_tokens = tokens[:, (-N * (i + 1) - 1) : (-N * (i) - 1)]
            labels = tokens[
                :,
                int(tokens.shape[1] + (-N * (i + 1))) : int(tokens.shape[1] + (-N * i)),
            ]
            string = tokenizer.decode(input_tokens[0])
            print("Decoded string from tokenizer: ", string)
            model_input = tokenizer(
                wolframe_prompt.replace("<REPLACEGPT>", string)+string,
                return_tensors="pt",
            )["input_ids"]
            # print(model_input.shape)
            with torch.no_grad():
                output = model(model_input.cuda()).logits.cpu()[:, -N:]
            new_outputs = self.generate_continuations(
                model_input, #tokenized cal prompt
                output,
                labels, #tokens of actual input
                model, #model of the model_inputs
                tokenizer,
                N,
                M
            )
            for output in new_outputs:
                if output is None:
                    continue
                output["index"] += int(tokens.shape[1] + (-N * (i + 1)))
                # filter by score
                if output["Score"] > 0.0:
                    outputs.append([output["Score"], output["index"]] + output["Wolframe_output"])
        return outputs