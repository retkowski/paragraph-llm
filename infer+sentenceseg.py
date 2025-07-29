import os

import nltk
import pandas as pd
import tabulate
import torch
from nltk import sent_tokenize
from termcolor import colored
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

nltk.download('punkt_tab')

DEBUG_MODE = True
SECTION_BASED_SEGMENTATION = False

model_id = "meta-llama/Llama-3.1-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

project_folder = "./"
partitions = ('train',)
file_prefix = "ytsegpara."
output_file_prefix = "ytsegpara-paragraphs."

data_dfs = {}
for partition in partitions:
    data_path = os.path.join(project_folder, file_prefix + partition + '.json')
    df = pd.read_json(data_path)
    
    if 'text' not in df.columns or 'targets' not in df.columns:
        raise ValueError(f"'text' or 'targets' column missing in {data_path}")
    
    data_dfs[partition] = df

system_message = "You are an AI assistant that helps users insert paragraphs into text."
user_message = ("""You are tasked with inserting paragraphs into a given text. The text will be provided to you, and your job is to break it up into coherent paragraphs. Here's the text you'll be working with:

{input}

Your task is to insert paragraph breaks into this text. A paragraph break should be signified by two newline characters.

A paragraph is a functionally or semantically coherent segment of text. This means that each paragraph should focus on a single main idea, topic, or function within the overall text. To identify where to insert paragraph breaks, consider the following guidelines:

1. Look for shifts in topic or focus
2. Identify transitions between different ideas or themes
3. Recognize changes in time, place, or perspective
4. Consider the length of the current segment (very long segments might benefit from being broken up)
5. Pay attention to transitional phrases or words that might signal a new paragraph

Please provide your final output with the inserted paragraph breaks. Ensure that you maintain the original text exactly as it was given, only adding the paragraph breaks where appropriate.""")

def get_messages(transcript, response):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": "Here is the segmentation of the video transcription into paragraphs:\n\n{response}"},
    ]
    messages[1]["content"] = messages[1]["content"].format(input=transcript)
    messages[2]["content"] = messages[2]["content"].format(response=response)
    return messages

def get_input_ids(transcript, response):
    input_ids = tokenizer.apply_chat_template(
        get_messages(transcript, response),
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt"
    ).to(model.device)
    input_ids = input_ids[:, :-1]
    return input_ids

def print_top_k(scores, k=3):
    top_ids = torch.topk(scores, k)
    indices = top_ids.indices[0]
    print_ids(scores, indices)

def print_ids(scores, ids):
    table = []
    for i, id in enumerate(ids):
        table.append((id, scores[0][id].item(), repr(tokenizer.decode(id))))
    print("\n")
    print(tabulate.tabulate(table, headers=["Token ID", "Score", "Token"]))
    print("\n")

punctuation_tokens = [
    (13, '.'), (30, '?'), (1, '"'), (0, "!"), (8, ')'),
    (497, '..'), (1131, '...'), (1210, '."'), (3001, '!!'), (12340, '!!!'), (7801, '??'),
    (570, ")."), (948, "]."), (662, ' .'), (949, ' ?'), (758, ' !'),
    (3343, '".'),
    (11453, '”.'), (27074, '?!'), (58490, '!?'), (10380, '?)'),
    (25750, '.]'), (94068, '?]'),
]
punctuation_ids = [id for id, _ in punctuation_tokens]
newline_tokens = [
    (382, '.\n\n'), (271, '\n\n'), (696, ')\n\n'), (1473, ':\n\n'), (1980, '?\n\n'), (2268, '!\n\n'),
    (2266, '."\n\n'),
    (1875, '"\n\n'), (2195, '...\n\n'), (6905, ' .\n\n'),
    (71291, '??\n\n'), (25833, '!!\n\n'),
]
newline_ids = [id for id, _ in newline_tokens]

def get_token_repr(token_id):
    return repr(tokenizer.decode(token_id)) + f" ({token_id})"

def extract_sections(sentences, targets):
    sections = []
    current_section = []
    for sentence, target in zip(sentences, targets):
        if target == '1' and current_section:
            sections.append(' '.join(current_section))
            current_section = []
        current_section.append(sentence)
    if current_section:
        sections.append(' '.join(current_section))
    return sections

for partition in partitions:
    df = data_dfs[partition]
    
    output_file = os.path.join(project_folder, f"{output_file_prefix}.{partition}.json")
    
    if os.path.exists(output_file):
        df_existing = pd.read_json(output_file, lines=True)
        df = df.merge(df_existing[['segmented_sections', 'paragraph_targets']], left_index=True, right_index=True, how='left')
    else:
        df['segmented_sections'] = None
        df['paragraph_targets'] = None
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {partition} partition"):
        segmented_sections = row.get('segmented_sections')

        if isinstance(segmented_sections, list) and len(segmented_sections) > 0:
            print(f"Skip {idx}")
            continue

        sentences = row['text']

        if not isinstance(sentences, list):
            df.at[idx, 'segmented_sections'] = []
            df.at[idx, 'paragraph_targets'] = ''
            continue

        if SECTION_BASED_SEGMENTATION:
            targets = row['targets'][2:]

            if not isinstance(targets, str) or len(sentences) != len(targets):
                df.at[idx, 'segmented_sections'] = []
                df.at[idx, 'paragraph_targets'] = ''
                continue

            sections = extract_sections(sentences, targets)

        else:
            sections = [' '.join(sentences)]

        segmented_sections = []
        all_sentences = []
        paragraph_targets = []

        for section_idx, section in enumerate(sections):
            section_sentences = sent_tokenize(section)
            if not section_sentences:
                segmented_sections.append([])
                continue

            past_key_values = DynamicCache()
            current_output = section_sentences[0]

            print(f"\nProcessing section {section_idx}/{len(sections)}")
            print(f"Number of sentences: {len(section_sentences)}")

            for i in range(len(section_sentences) - 1):
                next_sentence = " " + section_sentences[i + 1]
                expected_token_id = tokenizer(next_sentence, return_tensors="pt", add_special_tokens=False)["input_ids"][0, 0]

                input_ids = get_input_ids(section, current_output)
                input_length = input_ids.shape[1]

                if DEBUG_MODE:
                    print(f"\nSentence {i+1}")
                    print(f"Input sequence length: {input_length} tokens")

                if input_ids[0, -1] in punctuation_ids:
                    expected_token_id = input_ids[0, -1]
                    input_ids = input_ids[:, :-1]
                    current_output = current_output.removesuffix(punctuation_tokens[punctuation_ids.index(expected_token_id)][1])
                else:
                    print("Fallback: No recognized punctuation token found. Appending sentence directly.")
                    current_output += next_sentence
                    continue

                if len(past_key_values) == 0:
                    new_token_length = input_ids.shape[1]
                else:
                    cached_length = past_key_values[0][0].shape[2]
                    new_token_length = input_ids.shape[1] - cached_length

                if new_token_length <= 0:
                    print("No new tokens available; skipping generation for this sentence.")
                    current_output += next_sentence
                    continue
                
                try:
                    output = model.generate(
                        input_ids,
                        max_new_tokens=1,
                        do_sample=False,
                        temperature=1.0,
                        output_scores=True,
                        return_dict_in_generate=True,
                        past_key_values=past_key_values
                    )
                except RuntimeError:
                    breakpoint()

                scores = output['scores'][0]
                if DEBUG_MODE:
                    print_top_k(scores)
                    print_ids(scores, [expected_token_id] + newline_ids)

                    print(colored("Expected Token", attrs=["bold"]), colored(get_token_repr(expected_token_id), "cyan"))

                    print("-------- " + colored("Current Input", "light_green", attrs=["bold"]) +" -------")
                    print(colored(tokenizer.decode(input_ids[0]).split("<|start_header_id|>assistant<|end_header_id|>")[1].lstrip(), "light_green"))
                    print("------------------------------")

                    breakpoint()

                allowed_ids = torch.tensor(newline_ids + [expected_token_id]).to(model.device)
                scores_filtered = scores[0][allowed_ids]
                top_token = allowed_ids[torch.argmax(scores_filtered)]

                if DEBUG_MODE:
                    print(f"Top token: {tokenizer.decode(top_token.item())!r}")

                current_output += tokenizer.decode(top_token.item())
                if current_output[-1].isspace():
                    next_sentence = next_sentence[1:]
                current_output += next_sentence

            paragraphs = current_output.split('\n\n')
            section_paragraphs = []
            for paragraph in paragraphs:
                sentences_in_paragraph = sent_tokenize(paragraph)
                section_paragraphs.append(sentences_in_paragraph)
                for sentence in sentences_in_paragraph:
                    all_sentences.append(sentence)
                    paragraph_targets.append('0')
                if paragraph_targets:
                    paragraph_targets[-1] = '1'

            segmented_sections.append(section_paragraphs)

        df.at[idx, 'segmented_sections'] = segmented_sections
        df.at[idx, 'paragraph_targets'] = '|=' + ''.join(paragraph_targets)

        if idx % 1 == 0:
            print(f"Saving progress at index {idx}")
            df.to_json(output_file, orient='records', lines=True)
    
    df.to_json(output_file, orient='records', lines=True)
