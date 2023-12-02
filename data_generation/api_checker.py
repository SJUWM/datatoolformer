
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
import dateutil.parser as dparser
from calendar import IllegalMonthError
import random
import re

N=10

@dataclass
class AvailableAPIs:
    """Keeps track of available APIs"""

    retrieval: bool = True
    calendar: bool = True
    calculator: bool = True
    llmchain: bool = True
    wolframe: bool = True

    def check_any_available(self):
        return any([self.wolframe, self.calendar, self.calculator])


def check_apis_available(
    data: dict, tokenizer: PreTrainedTokenizerBase
) -> AvailableAPIs:
    """
    Returns available APIs with boolean flags

    :param data: from load_dataset, assumes ['text'] is available
    :param tokenizer: Tokenizer to tokenize data
    :return: AvailableAPIs
    """
    tokenized_data = tokenizer(data["text"])["input_ids"]
    available = AvailableAPIs()
    # In case we need a different version, found this here:
    # https://stackoverflow.com/questions/28198370/regex-for-validating-correct-input-for-calculator
    calc_pattern = re.compile("^(\d+[\+\-\*\/]{1})+\d+$")
    if len(tokenized_data) < 4096:
        available.retrieval = False
    try:
        # date = dparser.parse(data["text"], fuzzy=True)
        date = dparser.parse(data["text"], fuzzy=True)
        #print(date)
    except (ValueError, OverflowError, IllegalMonthError, TypeError):
        available.calendar = False
    available.calculator = False
    available.wolframe = False
    tried_rand = False
    if(len(tokenized_data) < 128):
        n=len(tokenized_data)
        print(n)
    for i in range(len(tokenized_data) // n):
        text = tokenizer.decode(tokenized_data[i * n : (i + 1) * n])
        print("text: ", text)

        operators = bool(re.search(calc_pattern, text))
        equals = any(
            ["=" in text, "equal to" in text, "total of" in text, "average of" in text, "How many"]
        )
        #print("\nAPI checker found equals: " + str(equals) + "operator: "+ str(operators))
        if not (operators or equals) and not tried_rand:
            tried_rand = True
            text = text.replace("\n", " ")
            text = text.split(" ")
            text = [item for item in text if item.replace(".", "", 1).isnumeric()]
            if len(text) >= 3:
                if random.randint(0, 99) == 0:
                    available.calculator = True
                    available.wolframe = True
        else:          
            available.calculator = True
            available.wolframe = True

    return available
