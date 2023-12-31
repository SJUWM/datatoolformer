import torch
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from tools import Calculator
from prompts import calculator_prompt
from typing import List
from data_generation.base_api import APICallPostprocessing
import dateutil.parser as dparser


# TODO: Per API?
MAX_BATCH_SIZE = 1 
N = 32  # SEQ Len
M = 16  # Min Loss Span To Consider
MAX_LEN = 1024  # Maximum calculator length


class CalculatorPostprocessing(APICallPostprocessing):
    def __init__(
        self,
        start_tokens: List[int],
        end_tokens: List[int],
        minimum_percentage: float = 0.0,
    ):
        self.calculator = Calculator
        self.api_text = "Calculator("
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
        # print("\n**in add api_calls**")
        # print("Texts to test: ", texts_to_test)
        # print("generated_text: ", outputs[0]["generated_text"])
        # print(outputs)
        # print("*****")
        for j in range(len(outputs)):
            outputs[j]["Calculator"] = outputs[j]["generated_text"].replace(
                texts_to_test[candidate], ""
            )
            # print("afte adding cacluclator key")
            # #print("Outputs dict: ")
            # print(outputs[j]["Calculator"])
            # print("*****")
            outputs[j]["Generated"] = outputs[j]["generated_text"].split("Output:")[-1]
            #print("Generated: ", outputs[j]["Generated"] )
            if "]" in outputs[j]["Calculator"]:
                outputs[j]["Calculator"] = (
                    outputs[j]["Calculator"].replace("Calculator(", "").split("]")[0]
                )
                if ")" in outputs[j]["Calculator"]:
                    outputs[j]["Calculator"] = outputs[j]["Calculator"].split(")")[0]
                outputs[j]["Calculator_text"] = (
                    "[Calculator(" + outputs[j]["Calculator"] + ")"
                )
                base_inputs = tokenizer(
                    outputs[j]["Calculator_text"] + "]" + "\n",
                    return_tensors="pt",
                )["input_ids"].cuda()
                #print("After split Calculator: ", (outputs[j]["Calculator"]))
                try:
                    outputs[j]["Calculator"] = self.calculator(outputs[j]["Calculator"])
                except (ValueError, TypeError, ZeroDivisionError):
                    continue
                if outputs[j]["Calculator"] is None:
                    continue
                outputs[j]["Calculator_output"] = [outputs[j]["Calculator_text"][1:], str(outputs[j]["Calculator"])]
                # print("Calculator_output")
                # print(outputs[j]["Calculator_output"])
                outputs[j]["Calculator_text"] = (
                    outputs[j]["Calculator_text"]
                    + "->"
                    + str(outputs[j]["Calculator"])
                    + "]"
                )
                test_inputs = tokenizer(
                    outputs[j]["Calculator_text"] + "\n",
                    return_tensors="pt",
                )["input_ids"].cuda()
                #print("Calculator_text after all modifications:",  outputs[j]["Calculator_text"])
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
        #print("***In Parse Article***")
        outputs = list()
        tokens = tokenizer(data["text"], return_tensors="pt")["input_ids"]
        global N
        N= tokens.shape[1] - 1
        global M
        # if N < M:
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
            #print("Decoded string from tokenizer: ", string)
            model_input = tokenizer(
                calculator_prompt.replace("<REPLACEGPT>", string)+string,
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
                    outputs.append([output["Score"], output["index"]] + output["Calculator_output"])
        return outputs
