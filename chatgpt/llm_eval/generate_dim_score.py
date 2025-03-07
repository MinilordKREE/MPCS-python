import argparse
import regex
import gzip
import json
import re
import math
import pandas as pd
from llm_utils import data_process, api_call

def separate_get_five_dims(
    prompt, 
    moral_stat,
    deployment_name,
    temperature,
    max_tokens,
    top_p,
    logprobs,
    top_logprobs,
    verbose
):
    dim_pair_list = \
    [("care", "Care/Harm: The Care/Harm foundation is rooted in the innate human capacity for\
                empathy and compassion towards others. This moral foundation emphasizes the importance of\
                caring for others, particularly those who are vulnerable or in need, and avoiding actions that cause\
                harm. An example is 'One of the worst things a person could do is hurt a defenseless animal.'"),
    
    ("fairness", "Fairness/Cheating: The Fairness/Cheating foundation is centered on the human inclination\
                        towards equitable treatment. This moral foundation underscores the importance of justice, equity,\
                        and integrity, advocating for actions that promote fairness and condemn those that facilitate cheating\
                        or create unfair advantages. An example is 'Justice is the most important requirement for a society.'"),
    
    ("loyalty", "Loyalty/Betrayal: The Loyalty/Betrayal foundation centers on the human tendency\
                        towards forming strong group affiliations and maintaining solidarity with those groups. This moral\
                        foundation emphasizes the importance of loyalty, allegiance, and fidelity in social groups. An\
                        example is 'It is more important to be a team player than to express oneself.'"),

    ("authority", "Authority/Subversion: The Authority/Subversion foundation revolves around the relationships between individuals and institutions that symbolize leadership and social hierarchy. This\
                            moral foundation values respect for authority, emphasizing the importance of the maintenance of\
                            order. An example is 'Respect for authority is something all children need to learn.'"),

    ("sanctity" ,"Sanctity/Degradation: The Sanctity/Degradation foundation is based on the concept\
                            of protecting the sacredness of life and the environment, which invokes a deep-seated disgust or\
                            contempt when these are degraded. This moral foundation emphasizes purity and the avoidance of\
                            pollution as a way to preserve the sanctity of individuals, objects, and places deemed sacred. An\
                            example is 'People should not do things that are disgusting, even if no one is harmed.'")]

    five_dim_dict = {"care": [], "fairness": [], "loyalty": [], "authority": [], "sanctity": []}
    five_dim_answer_dict = {"care": [], "fairness": [], "loyalty": [], "authority": [], "sanctity": []}
    if verbose:
        print(f"Statement: {moral_stat}")
    for dim, dim_pair in dim_pair_list:
        generation_prompt = prompt.format(text=moral_stat, dimension_pair = dim_pair)
        response = api_call.api_call(
                    generation_prompt, deployment_name, temperature, max_tokens, top_p, logprobs, top_logprobs
        )
        response_content = response.message.content

        print(response.message.content)
        print("end of reponse")

        if response_content.startswith("0") or response_content.startswith("1"):
            logprobs_items = response.logprobs.content

            # Filter for 0 and 1 tokens
            for logprob in logprobs_items:
                if logprob.token in ["0", "1"]:
                    if logprob.token == "0":
                        anti_token = "1"
                    elif logprob.token == "1":
                        anti_token = "0"
                    
                    tokens = [top.token for top in logprob.top_logprobs]
                    if anti_token in tokens:
                        # the anti token is in the top tokens, then calculate the probability with scaling such that 0 and 1 take up the whole thing
                        curr_poss = math.exp(logprob.logprob) / (math.exp(logprob.top_logprobs[tokens.index(anti_token)].logprob) + math.exp(logprob.logprob))
                    else:
                        # anti token not there, bump the probability to one
                        curr_poss = 1
                    token_prob_pair = (logprob.token, curr_poss)
                    dim_answer = logprob.token
            
            print(token_prob_pair)
            if token_prob_pair[0] == '0':
                confidence_score = 1 - token_prob_pair[1]
            else:
                confidence_score = token_prob_pair[1]
            five_dim_dict[dim] = confidence_score
            five_dim_answer_dict[dim] = dim_answer
    
    if verbose:
        print(five_dim_dict)

    return response, five_dim_dict, five_dim_answer_dict


def get_five_dims(
    prompt, 
    moral_stat,
    deployment_name,
    temperature,
    max_tokens,
    top_p,
    logprobs,
    top_logprobs,
    with_reason,
    verbose
):
    if verbose:
        print(f"Statement: {moral_stat}")
        print(with_reason)
    if with_reason == True:
        target_format = r'{\s*"Care/Harm":\s*\(\s*(0|1),\s*(.*?)\s*\),\s*"Fairness/Cheating":\s*\(\s*(0|1),\s*(.*?)\s*\),\s*"Loyalty/Betrayal":\s*\(\s*(0|1),\s*(.*?)\s*\),\s*"Authority/Subversion":\s*\(\s*(0|1),\s*(.*?)\s*\),\s*"Sanctity/Degradation":\s*\(\s*(0|1),\s*(.*?)\s*\),?.?\??\s*}'
    elif with_reason == "reason_first":
        target_format = r'{\s*"Care/Harm":\s*\(\s*(.*?)\s*,"? \s*(0|1)\),\s*"Fairness/Cheating":\s*\(\s*(.*?)\s*,"? \s*(0|1)\),\s*"Loyalty/Betrayal":\s*\(\s*(.*?)\s*,"? \s*(0|1)\),\s*"Authority/Subversion":\s*\(\s*(.*?)\s*,"? \s*(0|1)\),\s*"Sanctity/Degradation":\s*\(\s*(.*?)\s*,"? \s*(0|1)\),?.?\??\s*}'
    else:
        target_format = r'(?:```json\s*)?\{\s*"Care/Harm":\s*([01]),\s*"Fairness/Cheating":\s*([01]),\s*"Loyalty/Betrayal":\s*([01]),\s*"Authority/Subversion":\s*([01]),\s*"Sanctity/Degradation":\s*([01]),?.?\s*\}(?:\s*```)?'
    generation_prompt = prompt.format(text=moral_stat)

    response = api_call.api_call(
                generation_prompt, deployment_name, temperature, max_tokens, top_p, logprobs, top_logprobs
    )

    response_content = response.message.content

    print(response.message.content)
    print("end of reponse")

    five_dim_dict = {"care": [], "fairness": [], "loyalty": [], "authority": [], "sanctity": []}
    five_dim_answer_dict = {"care": [], "fairness": [], "loyalty": [], "authority": [], "sanctity": []}
    
    if regex.match(target_format, response_content):
        # Extract logprobs for 0 and 1
        logprobs_items = response.logprobs.content

        # Filter for 0 and 1 tokens
        token_prob_list = []
        five_dim_answer_list = []
        for logprob in logprobs_items:
            if logprob.token in ["0", "1"]:
                if logprob.token == "0":
                    anti_token = "1"
                elif logprob.token == "1":
                    anti_token = "0"
                
                tokens = [top.token for top in logprob.top_logprobs]
                if anti_token in tokens:
                    # the anti token is in the top tokens, then calculate the probability with scaling such that 0 and 1 take up the whole thing
                    curr_poss = math.exp(logprob.logprob) / (math.exp(logprob.top_logprobs[tokens.index(anti_token)].logprob) + math.exp(logprob.logprob))
                else:
                    # anti token not there, bump the probability to one
                    curr_poss = 1
                token_prob_list.append((logprob.token, curr_poss))
                five_dim_answer_list.append(logprob.token)
                
                    
        print(token_prob_list)
        confidence_score = []
        for token, prob in token_prob_list:
            if token == '0':
                confidence_score.append(1 - prob)
            else:
                confidence_score.append(prob)

        five_dim_dict["care"] = confidence_score[0]
        five_dim_dict["fairness"] = confidence_score[1]
        five_dim_dict["loyalty"] = confidence_score[2]
        five_dim_dict["authority"] = confidence_score[3]
        five_dim_dict["sanctity"] = confidence_score[4]

        five_dim_answer_dict["care"] = five_dim_answer_list[0]
        five_dim_answer_dict["fairness"] = five_dim_answer_list[1]
        five_dim_answer_dict["loyalty"] = five_dim_answer_list[2]
        five_dim_answer_dict["authority"] = five_dim_answer_list[3]
        five_dim_answer_dict["sanctity"] = five_dim_answer_list[4]

    if verbose:
        print(five_dim_dict)

    return response, five_dim_dict, five_dim_answer_dict



def can_convert_to_float(row):
    last_five = row.iloc[-5:]
    converted = pd.to_numeric(last_five, errors='coerce')  # Convert to numeric, invalid becomes NaN
    return not converted.isna().any()  # Return True if no NaN, False otherwise


def fix_non_float(result_file,
                  prompt, 
                    deployment_name,
                    temperature,
                    max_tokens,
                    top_p,
                    logprobs,
                    top_logprobs,
                    with_reason,
                    verbose):
    '''
    recursively fix those rows that don't have the outputs as float
    '''
    result_df = pd.read_csv(result_file)

    # non_float_mask = ~result_df.iloc[:, -5:].applymap(lambda x: isinstance(x, float)).all(axis=1)
    err_rows = result_df[~result_df.apply(can_convert_to_float, axis=1)]
    print(err_rows)
    while not err_rows.empty:
        for _, row in err_rows.iterrows():
            stat_id, stat = row["statement_id"], row["statement"]
            _ , five_dim_dict, _ = get_five_dims(
                prompt, 
                stat,
                deployment_name,
                temperature,
                max_tokens,
                top_p,
                logprobs,
                top_logprobs,
                with_reason,
                verbose
            )
            if all(isinstance(value, float) for value in five_dim_dict.values()):
                five_dim_dict["statement"] = stat
                five_dim_dict["statement_id"] = stat_id
                five_dim_dict = {"statement_id": five_dim_dict.pop("statement_id"), "statement": five_dim_dict.pop("statement"), **five_dim_dict}        
                result_df.loc[result_df['statement_id'] == stat_id, five_dim_dict.keys()] = five_dim_dict.values()
                err_rows = err_rows[err_rows["statement_id"] != stat_id]
    
    result_df.to_csv(result_file, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deployment_name",
        default="gpt-4o-mini",
        type=str,
        help="model",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=4096, help="max tokens to generate"
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="temperature")
    parser.add_argument("--top_p", type=float, default=0.0, help="top-p")
    parser.add_argument("--logprobs", type=float, default=True, help="logprobs")
    parser.add_argument("--top_logprobs", type=float, default=20, help="top_logprobs")
    parser.add_argument(
        "--input_file",
        type=str,
        default="/net/scratch2/chenxi/gpt/LLM4MFD/data/mftc_preprocessed.csv",
        help="data to run moral dimension extraction on",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="/net/scratch2/chenxi/gpt/LLM4MFD/prompts/baseline_prompt.txt",
        help="file to read prompts from",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="output/twitters/all/gpt4omini/gpt4omini_scores.csv",
        help="file to output score to",
    )
    parser.add_argument(
        "--answer_file",
        type=str,
        default="output/twitters/all/gpt4omini/gpt4omini_answers.csv",
        help="file to output model answers to",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="twitters"
    )
    parser.add_argument(
        "--verbose", type=bool, default=True, help="whether to print out results"
    )

    args = parser.parse_args()

    deployment_name, max_tokens, temperature, top_p, logprobs, top_logprobs= (
        args.deployment_name,
        args.max_tokens,
        args.temperature,
        args.top_p,
        args.logprobs,
        args.top_logprobs
    )
    if args.prompt_file.split('/')[-1] in ["whole_prompt.txt"]:
        with_reason = True
    elif args.prompt_file.split('/')[-1] in ["reason_first_prompt.txt", ""]:
        with_reason = "reason_first"
    else:
        with_reason = False

    prompt = open(args.prompt_file, "r").read()
    moral_stat_list, moral_stat_df = data_process.get_stats(args.input_file, args.mode)

    save_data = []
    answer_data = []

    for i, moral_stat in moral_stat_list:
        if args.prompt_file.split('/')[-1] in ["PNAS_prompt.txt"]:
            _ , five_dim_dict, five_answer_dict = separate_get_five_dims(
            prompt, 
            moral_stat,
            deployment_name,
            temperature,
            max_tokens,
            top_p,
            logprobs,
            top_logprobs,
            args.verbose
        )
        else:
            _ , five_dim_dict, five_answer_dict = get_five_dims(
                prompt, 
                moral_stat,
                deployment_name,
                temperature,
                max_tokens,
                top_p,
                logprobs,
                top_logprobs,
                with_reason,
                args.verbose
            )
        
        print(five_dim_dict)
        five_dim_dict["statement"] = moral_stat
        five_dim_dict["statement_id"] = i
        five_dim_dict = {"statement_id": five_dim_dict.pop("statement_id"), "statement": five_dim_dict.pop("statement"), **five_dim_dict}
        five_answer_dict = {"statement_id": i, "statement": moral_stat, **five_answer_dict}
        
        answer_data.append(five_answer_dict)
        save_data.append(five_dim_dict)

    save_df = pd.DataFrame(save_data)
    save_answer_df = pd.DataFrame(answer_data)

    save_df.to_csv(args.out_file, index=False)
    save_answer_df.to_csv(args.answer_file, index=False)

    # fix_non_float(args.out_file, prompt,
    #         deployment_name,
    #         temperature,
    #         max_tokens,
    #         top_p,
    #         logprobs,
    #         top_logprobs,
    #         with_reason,
    #         args.verbose)


if __name__ == "__main__":
    main()