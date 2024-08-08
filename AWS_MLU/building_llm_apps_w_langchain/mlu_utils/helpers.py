from langchain.schema import BaseOutputParser, OutputParserException
from typing import Any, Dict, List, Optional, Type, cast
from langchain.output_parsers.json import parse_and_check_json_markdown
import json


###########################################################
################# Validate Inference Parameters ###########
###########################################################

titan_models_inf_param = {
    "temperature": "(float) Temperature",
    "topP": "(float) Top P",
    "maxTokenCount": "(int) Response Length",
    "stopSequences": "([string]) Stop Sequences",
}


claude_models_inf_param = {
    "temperature": "(float) Temperature",
    "topP": "(float) Top P",
    "topK": "(float) Top K",
    "max_tokens_to_sample": "(int) Maximum length",
    "stop_sequences": "([string]) Stop sequences",
}



jurassic_models_inf_param = {
    "temperature": "(float) Temperature",
    "topP": "(float) Top P",
    "maxTokens": "(int) Maximum completion length",
    "stopSequences": "([string]) Stop sequences",
    "presencePenalty": "(float) Presence penalty",
    "countPenalty": "(int) Count penalty",
    "frequencyPenalty": "(int) Frequency penalty",
    "applyToWhitespaces": "(bool) Whitespaces penalty",
    "applyToPunctuation": "(bool) Punctuations penalty",
    "applyToNumbers": "(bool) Numbers penalty",
    "applyToStopwords": "(bool) Stop words penalty",
    "applyToEmojis" : "(bool) Emojis penalty",
    
}



command_models_inf_param = {
    "temperature": "(float) Temperature",
    "p": "(float) Top P",
    "k": "(float) Top K",
    "return_likelihoods" : "(string) Return likelihoods - GENERATION, ALL, NONE ",
    "stream": "(bool) Stream",
    "max_tokens": "(int) Maximum length",
    "stop_sequences": "(str) Stop sequences",
    "num_generations": "(int) Number of generations"
}

stability_models_inf_param = {
    "cfg_scale": "(float) Prompt Strength",
    "steps": "(int) Seed",
}

mixtral_models_inf_param = {
    "max_tokens" : "(int) Maximum completion length",
    "stop" : "(str) Stop sequences",    
    "temperature": "(float) Temperature",
    "top_p": "(float) Top P",
    "top_k": "(float) Top K"
}

def validate_inference_parameters(model_id, inference_parameters):    
    # Titan models
    if "titan" in model_id:    
        selection = titan_models_inf_param
    elif "ai21" in model_id:
        selection = jurassic_models_inf_param
    elif "claude" in model_id:
        selection = claude_models_inf_param
    elif "command" in model_id:
        selection = command_models_inf_param
    elif "stability" in model_id:
        selection = stability_models_inf_param
    elif "mixtral" in model_id:
        selection = mixtral_models_inf_param
    else:
        raise ValueError("Check the model id. \
        Currently we only support the following family of models:\
        Amazon - Titan, AI21 Labs - Jurassic, Anthropic - Claude, \
        Cohere - Command and Stability AI - Stable Diffusion")
    
    for key in inference_parameters:
        if key not in selection:
            raise ValueError("'{}' is not a valid inference paramater for model: {}".format(key, model_id))
    
    return True