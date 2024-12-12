import argparse
import random
from datasets import load_dataset
from transformers import pipeline, set_seed  # set seed for reproducibility
import torch
import re

set_seed(102)

model_name2model_id = {
    'Llama-3.1-8B-Instruct': "meta-llama/Meta-Llama-3.1-8B-Instruct"
}

def load_model(model_name):
    try:
        generator = pipeline(
            "text-generation",
            model=model_name2model_id[model_name],
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )
        return generator
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def format_prompt(prompt_type, story0, story1, model_name, examples=None, knowledge=None):
    if prompt_type == "zero-shot":
        system_prompt = f"""
        ## TASK 1
        Given the following two stories, identify which is the MORE PLAUSIBLE story. 
        If Story A is the MORE PLAUSIBLE story, include '#### 0' in the final line of your response. 
        If Story B is the MORE PLAUSIBLE story, include '#### 1' in the final line of your response.

        ## TASK 2
        For the story that is NOT PLAUSIBLE, identify ALL conflicting pairs of sentences.
        i, j, k are integers representing sentence numbers.
        If sentence i and j are conflicting, include '%%%% i, j' in the final line of your response, e.g. '%%%% 2, 5'
        If sentence i is in conflict with both sentence j and k, include '%%%% j, i %%%% k, i' in the final line of your response, e.g. '%%%% 1, 5 %%%% 2, 5'.
        Ensure that in a pair, the smaller integer is placed in front.
        """

        user_prompt = f"""
        Story A: {story0}\n
        Story B: {story1}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return messages

    elif prompt_type == "cot":
        system_prompt = f"""
        ## TASK 1
        Given the following two stories, identify which is the MORE PLAUSIBLE story. 
        If Story A is the MORE PLAUSIBLE story, include '#### 0' in the final line of your response. 
        If Story B is the MORE PLAUSIBLE story, include '#### 1' in the final line of your response.

        ## TASK 2
        For the story that is NOT PLAUSIBLE, identify ALL conflicting pairs of sentences.
        i, j, k are integers representing sentence numbers.
        If sentence i and j are conflicting, include '%%%% i, j' in the final line of your response, e.g. '%%%% 2, 5'
        If sentence i is in conflict with both sentence j and k, include '%%%% j, i %%%% k, i' in the final line of your response, e.g. '%%%% 1, 5 %%%% 2, 5'.
        Ensure that in a pair, the smaller integer is placed in front.
        """

        user_prompt = f"""
        Let's think step-by-step. 
        Carefully read each story sentence by sentence.
        Ientify sentences that contradict each other, create logical inconsistencies, or are implausible, based on the current physical state of objects.
        Justify why they contradict each other.
        Then, formulate your final answers.

        Story A: {story0}\n
        Story B: {story1}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return messages

    elif prompt_type == "cot_icl":
        system_prompt = f"""
        ## TASK 1
        Given the following two stories, identify which is the MORE PLAUSIBLE story. 
        If Story A is the MORE PLAUSIBLE story, include '#### 0' in the final line of your response. 
        If Story B is the MORE PLAUSIBLE story, include '#### 1' in the final line of your response.

        ## TASK 2
        For the story that is NOT PLAUSIBLE, identify ALL conflicting pairs of sentences.
        i, j, k are integers representing sentence numbers.
        If sentence i and j are conflicting, include '%%%% i, j' in the final line of your response, e.g. '%%%% 2, 5'
        If sentence i is in conflict with both sentence j and k, include '%%%% j, i %%%% k, i' in the final line of your response, e.g. '%%%% 1, 5 %%%% 2, 5'.
        Ensure that in a pair, the smaller integer is placed in front.
        """

        examples_prompt = ""
        for eg in examples:
            storyA, storyB = format_stories(eg['stories'])
            label = eg['label']
            confl_pairs = eg['confl_pairs']
            examples_prompt += f"Story A: {storyA}\nStory B: {storyB}\n\n #### {label}\n"
            for pair_list in confl_pairs:
                examples_prompt += f"%%%% {pair_list[0]}, {pair_list[1]}\n"

        user_prompt = f"""
        {examples_prompt}

        Let's think step-by-step. 
        Carefully read each story sentence by sentence.
        Ientify sentences that contradict each other, create logical inconsistencies, or are implausible, based on the current physical state of objects.
        Justify why they contradict each other.
        Then, formulate your final answers.

        Story A: {story0}\n
        Story B: {story1}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return messages

    elif prompt_type == "gen-know":
        know_s0 = knowledge[0]
        know_s1 = knowledge[1]

        system_prompt = f"""
        ## EXISTING KNOWLEDGE
        Story A: {know_s0}
        Story B: {know_s1}

        ## TASK 1
        Given the following two stories, identify which is the MORE PLAUSIBLE story. 
        If Story A is the MORE PLAUSIBLE story, include '#### 0' in the final line of your response. 
        If Story B is the MORE PLAUSIBLE story, include '#### 1' in the final line of your response.

        ## TASK 2
        For the story that is NOT PLAUSIBLE, identify ALL conflicting pairs of sentences.
        i, j, k are integers representing sentence numbers.
        If sentence i and j are conflicting, include '%%%% i, j' in the final line of your response, e.g. '%%%% 2, 5'
        If sentence i is in conflict with both sentence j and k, include '%%%% j, i %%%% k, i' in the final line of your response, e.g. '%%%% 1, 5 %%%% 2, 5'.
        Ensure that in a pair, the smaller integer is placed in front.
        """
        
        user_prompt = f"""
        Story A: {story0}\n
        Story B: {story1}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return messages
                
def format_stories(stories):
    story0 = stories[0]["sentences"]
    story1 = stories[1]["sentences"]

    story0_formatted = "\n".join([f"{i+1}. {sentence}" for i, sentence in enumerate(story0)])
    story1_formatted = "\n".join([f"{i+1}. {sentence}" for i, sentence in enumerate(story1)])

    print(story0_formatted + "\n" + story1_formatted)

    return story0_formatted, story1_formatted

def zero_shot_prompting(model, test_data, model_name):
    acc_predictions = []
    con_predictions = []

    for example in test_data:
        print(f"EXAMPLE: {example}\n")
        story0, story1 = format_stories(example["stories"])
        prompt = format_prompt("zero-shot", story0, story1, model_name)

        response = model(prompt, max_length=4096, num_return_sequences=1)[0]['generated_text']
        print("FULL_RESPONSE:", response)

        response = next((msg['content'] for msg in response if msg['role'] == 'assistant'), None)

        match_label = re.search(r"#### \b(0|1)\b", response)
        if match_label:
            number = match_label.group(1) 
            final_answer = int(number)
        else: 
            final_answer = "-"

        match_conf = re.findall(r"%%%% \b[1-6], [2-6]\b", response)
        conflicting_pairs = []

        if match_conf:
            for conf in match_conf:
                conf_idx = conf[5:].split(", ")
                conflicting_pairs.append([int(conf_idx[0]), int(conf_idx[1])])

        print(f"LABEL: {final_answer}\n")
        print(f"CONFLICTS: {conflicting_pairs}\n")

        acc_predictions.append(final_answer)
        con_predictions.append(conflicting_pairs)

    return acc_predictions, con_predictions

def cot_prompting(model, test_data, model_name):
    acc_predictions = []
    con_predictions = []

    for example in test_data:
        print(f"EXAMPLE: {example}\n")
        story0, story1 = format_stories(example["stories"])
        prompt = format_prompt("cot", story0, story1, model_name)

        response = model(prompt, max_length=4096, num_return_sequences=1)[0]['generated_text']
        print("FULL_RESPONSE:", response)

        response = next((msg['content'] for msg in response if msg['role'] == 'assistant'), None)

        match_label = re.search(r"#### \b(0|1)\b", response)
        if match_label:
            number = match_label.group(1) 
            final_answer = int(number)
        else: 
            final_answer = "-"

        match_conf = re.findall(r"%%%% \b[1-6], [2-6]\b", response)
        conflicting_pairs = []

        if match_conf:
            for conf in match_conf:
                conf_idx = conf[5:].split(", ")
                conflicting_pairs.append([int(conf_idx[0]), int(conf_idx[1])])

        print(f"LABEL: {final_answer}\n")
        print(f"CONFLICTS: {conflicting_pairs}\n")

        acc_predictions.append(final_answer)
        con_predictions.append(conflicting_pairs)

    return acc_predictions, con_predictions

def cot_icl_prompting(model, test_data, model_name, examples):
    acc_predictions = []
    con_predictions = []

    for example in test_data:
        print(f"EXAMPLE: {example}\n")
        story0, story1 = format_stories(example["stories"])
        prompt = format_prompt("cot_icl", story0, story1, model_name, examples=examples)

        response = model(prompt, max_length=4096, num_return_sequences=1)[0]['generated_text']
        print("FULL_RESPONSE:", response)

        response = next((msg['content'] for msg in response if msg['role'] == 'assistant'), None)

        match_label = re.search(r"#### \b(0|1)\b", response)
        if match_label:
            number = match_label.group(1) 
            final_answer = int(number)
        else: 
            final_answer = "-"

        match_conf = re.findall(r"%%%% \b[1-6], [2-6]\b", response)
        conflicting_pairs = []

        if match_conf:
            for conf in match_conf:
                conf_idx = conf[5:].split(", ")
                conflicting_pairs.append([int(conf_idx[0]), int(conf_idx[1])])

        print(f"LABEL: {final_answer}\n")
        print(f"CONFLICTS: {conflicting_pairs}\n")

        acc_predictions.append(final_answer)
        con_predictions.append(conflicting_pairs)

    return acc_predictions, con_predictions

def generate_knowledge(model, story0, story1):
    SYSTEM_PROMPT = f"""
    Generate knowledge on the impact of an action on the physical state of objects and the plausibility of the action. Examples:
    Input: 
    1. John played on the swings.\n
    2. John went towards the bench.\n
    3. John opened his bag but found that he had left his water bottle at home.\
    4. John drank from his water bottle.\n
    5. John passed the bottle to his friend.\n    
    Analysis: 
    1. The swings exist and are working.\n
    2. John stopped playing on the swings. John is at the bench.\n
    3. The bag is present. The water bottle is not present. John is unable to drink water from his water bottle.\n
    4. This is NOT possible. The water bottle is not present.\n
    
    Input:
    1. Mary turned on the table lamp.\n
    2. Mary took out her pencil and notebook.\n
    3. Mary drew heart shapes on all the pages of her notebook with a pencil.\n
    4. Mary cut out the heart shaped drawings with her scissors.\n
    5. Mary wrote a letter to her friend on her notebook.\n
    Analysis:
    1. The lamp is working and turned on.\n
    2. The pencil and notebook are present.\n
    3. The notebook is filled. There are no empty pages in the notebook. There are heart shaped drawings on all the pages.\n
    4. The heart shaped drawings on every page is cut out.\n
    5. This is NOT possible. There are no empty pages in the notebook.\n

    Input:
    1. Mary put her shorts and a tennis ball into a duffle bag.\n
    2. Mary grabbed the bag and threw it in the trash.\n
    3. Mary changed into her shorts and went outside to bounce the tennis ball.\n
    4. Mary bounced the tennis ball over the fence and could not reach it.\n
    5. Mary shrugged and went home.\n
    Analysis:
    1. The duffle bag contains Mary's shorts and a tennis ball.\n
    2. The duffle bag is in the trash. Mary's shorts and the tennis ball are in the trash.\n
    3. This is NOT possible. Mary's shorts and the tennis ball are in the trash.\n
    """ 
    USER_PROMPT_S0 = f"""
    Generate the analysis for the following input.
    Input:
    {story0}
    Analysis:
    """
    USER_PROMPT_S1 = f"""
    Generate the analysis for the following input.
    Input:
    {story1}
    Analysis:
    """ 
    messages_s0 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_S0},
    ]
    messages_s1 = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_S1},
    ]
    s0_know = model(messages_s0, max_length=4096, num_return_sequences=1)[0]['generated_text']
    s1_know = model(messages_s1, max_length=4096, num_return_sequences=1)[0]['generated_text']

    s0_know = next((msg['content'] for msg in s0_know if msg['role'] == 'assistant'), None)
    s1_know = next((msg['content'] for msg in s1_know if msg['role'] == 'assistant'), None)

    return s0_know, s1_know

def gen_know_prompting(model, test_data, model_name):
    acc_predictions = []
    con_predictions = []

    for example in test_data:
        print(f"EXAMPLE: {example}\n")
        story0, story1 = format_stories(example["stories"])
        story0_know, story1_know = generate_knowledge(model, story0, story1)
        knowledge = [story0_know, story1_know]
        prompt = format_prompt("gen-know", story0, story1, model_name, knowledge=knowledge)

        response = model(prompt, max_length=4096, num_return_sequences=1)[0]['generated_text']
        print("FULL_RESPONSE:", response)
        response = next((msg['content'] for msg in response if msg['role'] == 'assistant'), None)

        match_label = re.search(r"#### \b(0|1)\b", response)
        if match_label:
            number = match_label.group(1) 
            final_answer = int(number)
        else: 
            final_answer = "-"

        match_conf = re.findall(r"%%%% \b[1-6], [2-6]\b", response)
        conflicting_pairs = []

        if match_conf:
            for conf in match_conf:
                conf_idx = conf[5:].split(", ")
                conflicting_pairs.append([int(conf_idx[0]), int(conf_idx[1])])

        print(f"LABEL: {final_answer}\n")
        print(f"CONFLICTS: {conflicting_pairs}\n")

        acc_predictions.append(final_answer)
        con_predictions.append(conflicting_pairs)

    return acc_predictions, con_predictions

def evaluate_predictions(predictions, ground_truth):
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return correct / len(predictions) if predictions else 0

def main():
    dataset = load_dataset("sled-umich/TRIP", split="ClozeTest")
    model_name = "Llama-3.1-8B-Instruct"

    # method = "zero-shot"
    # method = "cot"
    # method = "cot_icl"
    method = "gen-kno"
    num_examples = 0

    model = load_model(model_name)
    if model is None:
        return

    examples = []
    if num_examples > 0:
        train_data = load_dataset("sled-umich/TRIP", split="ClozeTrain")
        examples = random.sample(list(train_data), min(num_examples, len(train_data)))

    # Run appropriate prompting method
    if method == 'zero-shot':
        acc_predictions, con_predictions = zero_shot_prompting(model, dataset, model_name)
    elif method == 'cot':
        acc_predictions, con_predictions = cot_prompting(model, dataset, model_name)
    elif method == 'cot_icl':
        acc_predictions, con_predictions = cot_prompting(model, dataset, model_name, examples)
    else:  # gen-know
        acc_predictions, con_predictions = gen_know_prompting(model, dataset, model_name ,)

    # Calculate and print accuracy
    acc_ground_truth = [example["label"] for example in dataset]
    print("PREDICTIONS\n", acc_predictions)
    print("GROUND_TRUTH\n", acc_ground_truth)
    accuracy = evaluate_predictions(acc_predictions, acc_ground_truth)
    print(f"Accuracy: {accuracy:.2%}")

    con_ground_truth = [example["confl_pairs"] for example in dataset]
    print("PREDICTIONS\n", con_predictions)
    print("GROUND_TRUTH\n", con_ground_truth)
    consistency = evaluate_predictions(con_predictions, con_ground_truth)
    print(f"Consistency: {consistency:.2%}")

if __name__ == "__main__":
    main()