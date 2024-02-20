from ray import tune
from .shared import project_root

# Ray Tune Configs
rt_config = {
    "batch_size": tune.choice([16, 32, 64]),
    "epochs": tune.uniform(5, 20),
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "cyclic_loss_weight": tune.loguniform(1e-4, 1e-1),
    "optimizer": tune.grid_search(["Adam", "SGD", "RMSprop"]),
    "hidden_size": tune.choice([64, 128, 256, 512, 1024]),
    "num_layers": tune.grid_search([1, 2, 4, 8]),
}
rt_num_samples = 10
rt_resources_per_trial = {"cpu": 1, "gpu": 0}
result_grid_path = project_root / "data/historical/train/ray_tune_results/"
result_grid_exp_name = "FinMAF_ray_tune_results"
