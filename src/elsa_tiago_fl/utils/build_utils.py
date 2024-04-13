import torch


def build_model(config):
    available_models = ["dqn","ddpg"]
    multimodal = check_types(config)

    #input('Wanna build {:} with mutimodal = {:}?'.format(config.model_name.lower(), config.multimodal))

    if not config.model_name.lower() in available_models:
        raise ValueError(
                "Value '{:s}' for model selection not expected. Please choose one of {:}".format(
                    config.model_name, ", ".join(available_models)
                )
            )

    if config.model_name.lower() == "dqn":
        from models.dqn import DQN
        model = DQN(
            input_dim=config.input_dim,
            action_dim=config.action_dim,
            device="cuda",
            config=config,
            multimodal=multimodal,
        )
    elif config.model_name.lower() == "ddpg":
        from models.ddpg import DDPG
        model = DDPG(
            input_dim=config.input_dim,
            action_dim=config.action_dim,
            device="cuda",
            config=config,
            multimodal=multimodal,
        )
    else:
        raise Exception
    model.to(config.device)

    return model


def build_optimizer(model, config):
    optimizer_class = getattr(torch.optim, "Adam")
    parameters = model.optimizer_params
    return optimizer_class(parameters, lr = float(config.lr))
        
#    return optimizer_class(model.parameters(), lr=float(config.lr))

def check_types(config):
    if not(type(config.input_dim)==list or type(config.input_dim)==int):
        raise ValueError(
        "Value type '{:s}' for input dimention not expected. Please choose one of int or list".format(
            str(type(config.input_dim)))
        )
    if not(type(config.action_dim)==list or type(config.action_dim)==int):
        raise ValueError(
        "Value type '{:s}' for action dimention not expected. Please choose one of int or list".format(
            str(type(config.action_dim)))
        )
    multimodal = True if type(config.input_dim) == list else False
    return multimodal