import torch


def build_model(config):
    available_models = ["dqn", "dqn_new", "ppo", "ppo_continue"]
    multimodal = check_types(config)

    #input('Wanna build {:} with mutimodal = {:}?'.format(config.model_name.lower(), config.multimodal))

    if not config.model_name.lower() in available_models:
        raise ValueError(
                "Value '{:s}' for model selection not expected. Please choose one of {:}".format(
                    config.model_name, ", ".join(available_models)
                )
            )

    match config.model_name.lower():
        case "dqn":
            from models.dqn import DQN
            model = DQN(
                input_dim=config.input_dim,
                action_dim=config.action_dim,
                device="cuda",
                config=config,
                multimodal=multimodal,
            )

        case "dqn_new":
            from models.dqn_new import DQN
            model = DQN(
                input_dim=config.input_dim,
                action_dim=config.action_dim,
                device="cuda",
                config=config,
                multimodal=multimodal,
                executable_act = config.executable_act
            )

        case "ppo":
            from models.ppo import PPO
            model = PPO(
                input_dim=config.input_dim,
                action_dim=config.action_dim,
                device="cuda",
                config=config,
                multimodal=multimodal,
            )
            
        case "ppo_continue":
            from models.ppo_continue import PPO
            net_width = config.net_width_multi if config.multimodal else config.net_width
            model = PPO(
                input_dim=config.input_dim,
                action_dim=config.action_dim,
                net_width=net_width,
                device="cuda",
                config=config,
                multimodal=config.multimodal,
            )

    model.to(config.device)

    return model


def build_optimizer(model, config):
    optimizer_class = getattr(torch.optim, "Adam")
    return optimizer_class(model.parameters(), lr=float(config.lr))

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