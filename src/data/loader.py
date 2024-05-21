import os
import shutil
import json
from tqdm import tqdm
from datasets import load_dataset
import datasets

dic = {
    0: 'negetive',
    1: 'positive',
    2: 'neutral',
}

def tfns_train_data():
    tfns = load_dataset('zeroshot/twitter-financial-news-sentiment')
    tfns = tfns['train']
    tfns = tfns.to_pandas()
    tfns['label'] = tfns['label'].apply(lambda x: dic[x])
    tfns['instruction'] = 'What is the sentiment of the following tweet? Please choose one from {negetive, positive, neutral}'
    tfns = tfns.rename(columns={'text': 'input', 'label': 'output'})
    tfns = datasets.Dataset.from_pandas(tfns)
    train_dataset = datasets.concatenate_datasets([tfns]*2)
    all_dataset = train_dataset.shuffle(42)

    return all_dataset

def save_dataset(dataset, path):
    data_list = []
    for item in dataset.to_pandas().itertuples():
        tmp = {}
        tmp['instruction'] = item.instruction
        tmp['input'] = item.input
        tmp['output'] = item.output
        data_list.append(tmp)

    with open(path, 'w') as f:
        for example in tqdm(data_list, desc='Saving and Formatting dataset'):
            context = f"Instruction: {example['instruction']}\n"
            if example.get('input'):
                context += f"Input: {example['input']}\n"
            context += "Answer: "
            target = example['output']
            f.write(json.dumps({'context': context, 'target': target}) + '\n')
    