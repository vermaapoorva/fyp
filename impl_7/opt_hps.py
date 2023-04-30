""" Optuna example that optimizes the hyperparameters of
a reinforcement learning agent using PPO implementation from Stable-Baselines3
on an OpenAI Gym environment.

This is a simplified version of what can be found in https://github.com/DLR-RM/rl-baselines3-zoo.

You can run this example as follows:
    $ python sb3_simple.py

"""
from typing import Any
from typing import Dict

import gymnasium as gym
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch
import torch.nn as nn

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

import custom_impl_7

N_TRIALS = 100
N_STARTUP_TRIALS = 0
N_EVALUATIONS = 5
N_TIMESTEPS = int(1e3)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 2

ENV_ID = "RobotEnv7-v0"

DEFAULT_HYPERPARAMS = {
    "policy": "CnnPolicy",
}

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for PPO hyperparameters."""
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    gae_lambda = 1.0 - trial.suggest_float("gae_lambda", 0.001, 0.2, log=True)
    n_steps = 2 ** trial.suggest_int("exponent_n_steps", 8, 14)
    learning_rate = trial.suggest_float("lr", 5e-6, 0.003)
    batch_size = 2 ** trial.suggest_int("exponent_batch_size", 11, 11)
    print("batch_size", batch_size)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    net_arch = trial.suggest_categorical("net_arch", ["large"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    
    # Display true values.
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("gae_lambda_", gae_lambda)
    trial.set_user_attr("n_steps", n_steps)

    if net_arch == "small":
        net_arch = [{"pi": [128, 128, 128], "vf": [128, 128, 128]}]
    elif net_arch == "medium":
        net_arch = [{"pi": [128, 256, 256, 128], "vf": [128, 256, 256, 128]}]
    else:
        net_arch = dict(pi=[128, 256, 512, 256, 128], vf=[128, 256, 512, 256, 128])

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    params = {
#        "n_steps": n_steps,
#        "gamma": gamma,
#        "gae_lambda": gae_lambda,
#        "learning_rate": learning_rate,
        "batch_size": batch_size,
#        "ent_coef": ent_coef,
#        "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            "net_arch": net_arch,
#            "activation_fn": activation_fn,
#            "ortho_init": ortho_init,
        },
    }

    print("Params: ", params)

    return params

class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial) -> float:
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters.
    kwargs.update(sample_ppo_params(trial))
    # Create the RL model.

    #env = make_vec_env(ENV_ID, n_envs=1, vec_env_cls=SubprocVecEnv)
    env = SubprocVecEnv([lambda : gym.make('RobotEnv7-v0') for _ in range(2)])

    model = PPO(env = env, **kwargs)
    # Create env used for evaluation.
    # eval_env = Monitor(gym.make(ENV_ID))
    log_dir = "logs_hp"
    eval_env = SubprocVecEnv([lambda : Monitor(gym.make('RobotEnv7-v0'), log_dir) for _ in range(1)])

    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvalCallback(
        eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
    )

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback, progress_bar=True)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training.
    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=600)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))
