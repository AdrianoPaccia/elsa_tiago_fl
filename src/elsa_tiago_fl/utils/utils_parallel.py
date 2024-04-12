import torch
from collections import OrderedDict
import os
import glob
import torch

""" Deprecated functions: They will send and load all the model's weights.
def get_parameters_from_model(model):
    return [val.cpu().numpy() for _, val in model.model.state_dict().items()]

def set_parameters_model(model, parameters):
    params_dict = zip(model.model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.model.load_state_dict(state_dict, strict=True)
"""


# New functions. They will send and load only weights of NON-frozen layers.
def get_parameters_from_model(model):
    keyed_parameters = {n: p.requires_grad for n, p in model.named_parameters()}
    frozen_parameters = [
        not keyed_parameters[n] if n in keyed_parameters else False
        for n, p in model.state_dict().items()
    ]
    return [
        val.cpu().numpy()
        for val, is_frozen in zip(model.state_dict().values(), frozen_parameters)
        if not is_frozen
    ]


def set_parameters_model(model, parameters):
    keyed_parameters = {n: p.requires_grad for n, p in model.named_parameters()}
    frozen_parameters = [
        not keyed_parameters[n] if n in keyed_parameters else False
        for n, p in model.state_dict().items()
    ]

    i = 0
    params_dict = model.state_dict()
    for key, is_frozen in zip(model.state_dict().keys(), frozen_parameters):
        # Update state dict with new params.
        if not is_frozen:
            params_dict[key] = torch.Tensor(parameters[i])
            i += 1

    model.load_state_dict(params_dict, strict=True)


def weighted_average(metrics_dict):
    metrics = list(metrics_dict[0][1].keys())
    aggregated_metrics_dict = {}
    dataset_length = sum([num_samples for num_samples, _ in metrics_dict])
    for metric in metrics:
        aggregated_metrics_dict[metric] = sum(
            [
                m[metric] * num_examples / dataset_length
                for num_examples, m in metrics_dict
            ]
        )

    return aggregated_metrics_dict



def save_weigths(model,reward,config,folder):
    save_dir = os.path.join(config.save_dir, 'weigths',folder)
    os.makedirs(save_dir, exist_ok=True)
    save_name = os.path.join(save_dir,
        config.model_name + '_' + config.env_name  
        )
    if config.multimodal:
        save_name = save_name + '_multimodal'
    else:
        save_name = save_name + '_input' + str(config.input_dim)
    save_name = save_name + '_'+ str(int(reward)) + '.pth'
    torch.save(model.state_dict(), save_name)
    return save_name

def get_model_with_highest_score(model,config, from_all=True):

    # Construct the pattern to match saved model files
    directory = os.path.join(config.save_dir, 'weigths')

    if from_all:
        folders = [os.path.join(directory,folder) for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]
        patterns = []
        for f in folders:
            pattern =os.path.join(f,f"{config.model_name}_{config.env_name}")
            # os.path.join(f,'client'+str(config.client_id),f"{config.model_name}_{config.env_name}")
            if config.multimodal:
                pattern += '_multimodal'
            else:
                pattern += f'_input{config.input_dim}'
            pattern += '_*.pth'
            patterns.append(pattern)
    else:
        patterns = [os.path.join(directory, "all_clients",f"{config.model_name}_{config.env_name}")]
    # List all matching model files
      
    winner_score = -9999
    winner_model = None
    try: 
        for pattern in patterns:
            model_files = glob.glob(pattern)
            scores = [float(os.path.basename(file).split('_')[-1].split('.pth')[0]) for file in model_files]
            if len(scores)>0:
                score_i = int(max(scores))
                if score_i > winner_score:
                    score_i = max(scores)
                    winner_score = score_i
                    winner_model = model_files[scores.index(score_i)]
        
            model.load_state_dict(torch.load(winner_model))
            print(f'Loaded the model parameters: {winner_model}')
            return model
    except:
        print('No model is available')
        return model

    #except Exception as e:
    #    raise RuntimeError("No model parameters available!")