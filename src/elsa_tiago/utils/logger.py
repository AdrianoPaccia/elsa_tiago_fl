import os, socket, datetime
import wandb as wb

import torch

# import wandb as wb
from elsa_tiago.utils.utils import Singleton


class Logger(metaclass=Singleton):
    def __init__(self, config):
        self.log_folder = config.save_dir
        self.experiment_name = config.experiment_name
        self.comms_log_file = os.path.join(
            self.log_folder,
            "communication_logs",
            "{:}.csv".format(self.experiment_name),
        )

        machine_dict = {"cvc117": "Local", "cudahpc16": "DAG", "cudahpc25": "DAG-A40"}
        machine = machine_dict.get(socket.gethostname(), socket.gethostname())

        env = config.env_name
        tags = [config.model_name, env, machine]
        log_config = {
            "Model": config.model_name,
            "Env": env,
            "Batch size": config.batch_size,
            "lr": config.lr,
            "seed": config.seed,
        }

        tags.append("FL Flower")

        log_config.update(
            {
                "FL Flower": True,
                "Sample Clients": config.fl_parameters.sample_clients,
                "Total Clients": config.fl_parameters.total_clients,
                "FL Rounds": config.fl_parameters.num_rounds,
                "Iterations per FL Round": config.fl_parameters.iterations_per_fl_round,
            }
        )

        self.logger = wb.init(
            project="Elsa_Robotics_FL_test",
            name=self.experiment_name,
            dir=self.log_folder,
            tags=tags,
            config=log_config,
        )
        self.logger.define_metric("Train/FL Round *", step_metric="fl_round")
        self._print_config(log_config)

        self.current_epoch = 0
        self.len_dataset = 0

    def _print_config(self, config):
        print("{:s} \n{{".format(config["Model"]))
        for k, v in config.items():
            if k != "Model":
                print("\t{:}: {:}".format(k, v))
        print("}\n")

    def log_model_parameters(self, model):
        total_params = 0
        trainable_params = 0
        for attr in dir(model):
            if isinstance(getattr(model, attr), torch.nn.Module):
                total_params += sum(
                    p.numel() for p in getattr(model, attr).parameters()
                )
                trainable_params += sum(
                    p.numel()
                    for p in getattr(model, attr).parameters()
                    if p.requires_grad
                )

        self.logger.config.update(
            {
                "Model Params": int(total_params / 1e6),  # In millions
                "Model Trainable Params": int(trainable_params / 1e6),  # In millions
            }
        )

        print(
            "Model parameters: {:d} - Trainable: {:d} ({:2.2f}%)".format(
                total_params, trainable_params, trainable_params / total_params * 100
            )
        )

    def log_val_metrics(
        self,
        avg_reward,
        std_reward,
        avg_episode_length,
        std_episode_length,
        update_best=False,
    ):
        str_msg = (
            f"FL Round {self.current_epoch}: Reward = {avg_reward} +- {std_reward} "
            f"| Episode length = {avg_episode_length} +- {std_episode_length}"
        )

        if self.logger:
            self.logger.log(
                {
                    "Val/FL Round Average Reward": avg_reward,
                    "Val/FL Round Std Reward": std_reward,
                    "Val/FL Round Average Episode Length": avg_episode_length,
                    "Val/FL Round Std Episode Length": std_episode_length,
                    "fl_round": self.current_epoch,
                }
            )

            if update_best:
                str_msg += "\tBest Average Reward!"
                self.logger.config.update(
                    {
                        "Best Average Reward": avg_reward,
                        "Best FL Round": self.current_epoch,
                    },
                    allow_val_change=True,
                )

        print(str_msg)
