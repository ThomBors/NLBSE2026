import logging
import random

import numpy as np
import torch


def tokenize_function(examples, tokenizer):
    result = tokenizer(examples["combo"])
    if tokenizer.is_fast:
        result["word_ids"] = [
            result.word_ids(i) for i in range(len(result["input_ids"]))
        ]
    return result


# def tokenize_function(batch, tokenizer):
#     # batch["combo"] is a list of strings
#     result = tokenizer(
#         batch["combo"],
#         padding="max_length",
#         truncation=True,
#         return_tensors=None  # Let datasets handle list outputs
#     )

#     # word_ids only works for fast tokenizer and single encoding at a time
#     if tokenizer.is_fast:
#         # create word_ids for each sentence
#         result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]

#     return result


def group_texts(examples, chunk_size=64):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result


def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def title():
    logging.info(
        """
        
 ███████╗██╗   ██╗███╗   ██╗████████╗██╗  ██╗███████╗████████╗██╗ ██████╗    
 ██╔════╝╚██╗ ██╔╝████╗  ██║╚══██╔══╝██║  ██║██╔════╝╚══██╔══╝██║██╔════╝    
 ███████╗ ╚████╔╝ ██╔██╗ ██║   ██║   ███████║█████╗     ██║   ██║██║        
 ╚════██║  ╚██╔╝  ██║╚██╗██║   ██║   ██╔══██║██╔══╝     ██║   ██║██║        
 ███████║   ██║   ██║ ╚████║   ██║   ██║  ██║███████╗   ██║   ██║╚██████╗   
 ╚══════╝   ╚═╝   ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝ ╚═════╝     

  ██████╗ ██╗   ██╗███████╗██████╗ ███████╗ █████╗ ███╗   ███╗██████╗ ██╗     ██╗███╗   ██╗ ██████╗                  
 ██╔═══██╗██║   ██║██╔════╝██╔══██╗██╔════╝██╔══██╗████╗ ████║██╔══██╗██║     ██║████╗  ██║██╔════╝          
 ██║   ██║██║   ██║█████╗  ██████╔╝███████╗███████║██╔████╔██║██████╔╝██║     ██║██╔██╗ ██║██║  ███╗  
 ██║   ██║╚██╗ ██╔╝██╔══╝  ██╔══██╗╚════██║██╔══██║██║╚██╔╝██║██╔═══╝ ██║     ██║██║╚██╗██║██║   ██║         
 ╚██████╔╝ ╚████╔╝ ███████╗██║  ██║███████║██║  ██║██║ ╚═╝ ██║██║     ███████╗██║██║ ╚████║╚██████╔╝         
  ╚═════╝   ╚═══╝  ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝     ╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝     
                                                                          
    """
    )
