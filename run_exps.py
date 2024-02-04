import os
import pickle
import time

# Save an integer to a pickle file
def save_to_pickle_file(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# Read an integer from a pickle file
def read_from_pickle_file(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


log_dir='single_exp_logs'
model='codellama'
# for task in ['cifar10', 'imdb', 'CLRS']:
# for task in ['imdb', 'CLRS', 'house-price', 'spaceship-titanic', 'feedback']:
# for task in ['amp-parkinsons-disease-progression-prediction']:
# for task in ['llama-inference', 'vectorization', 'babylm']:
for task in ['vectorization', 'babylm', 'bibtex-generation', 'literature-review-tool']:
    folder = f'{log_dir}/{model}/{task}'
    pklname = f'{log_dir}-{model}-{task}-done_count.pkl'

    if os.path.exists(pklname):
        i = read_from_pickle_file(pklname)
    else:
        i = 0

    while i < 8:
        
        folder_i = f'{folder}/{i}'
        workspaces_folder_i = f'workspaces/{folder}/{i}'

        if os.path.exists(folder_i): os.system(f'rm -rf {folder_i}')
        if os.path.exists(workspaces_folder_i): os.system(f'rm -rf {workspaces_folder_i}')

        os.makedirs(folder_i)
        os.makedirs(workspaces_folder_i)

        command = f'python -u -m MLAgentBench.runner --task {task} --device 0 --log-dir {folder_i}  --work-dir {workspaces_folder_i} --llm-name {model} --edit-script-llm-name {model} --fast-llm-name {model} > {folder}/{i}/log 2>&1'
        print(command)
        os.system(command)
        print('\n\n ------- DONE FOR i = ', i, ' ------------------\n\n')
        i+=1
        save_to_pickle_file(i, pklname)

    os.remove(pklname)