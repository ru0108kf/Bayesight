import ghhops_server as hs
from flask import Flask
import os
from typing import Optional
import warnings
import json
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import japanize_matplotlib
import filepath


# =================数値設定=================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

BATCH_SIZE = 3
NUM_RESTARTS = 10
RAW_SAMPLES = 512
bounds = torch.tensor([[0.0] * 6, [1.0] * 6], device=device, dtype=dtype)

# ファイルパス
SAVE_FOLDER_PATH = r'C:\Users\msy-t\ScriptFile\kansei\multiObjective\save_folder'
TENSOR_X_DATA_PATH = os.path.join(SAVE_FOLDER_PATH, 'tensor_X_data.pt')
JSON_Y_DATA_PATH = os.path.join(SAVE_FOLDER_PATH, 'json_Y_data.json')
TENSOR_CON_DATA_PATH = os.path.join(SAVE_FOLDER_PATH, 'tensor_con_data.pt')
MODEL_PATH = os.path.join(SAVE_FOLDER_PATH, 'model.pth')
MODEL_OBJECT_PATH = os.path.join(SAVE_FOLDER_PATH, 'model_object.pth')
IMG_PATH = os.path.join(SAVE_FOLDER_PATH, 'img')



# =================数値設定ここまで=================
# =================botorch実装部分=================
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from torch.quasirandom import SobolEngine
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf
from botorch import fit_gpytorch_mll
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler
import gpytorch
import numpy as np
import random

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

NOISE_SE = 0.25
train_yvar = torch.tensor(NOISE_SE**2, device=device, dtype=dtype)

seed = 7777
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def get_initial_points(dim, n_pts):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=7777)
    X = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X

def singletask_model(train_x, train_obj, train_con, state_dict=None):
    # define models for objective and constraint
    model_obj = SingleTaskGP(
        train_x,
        train_obj,
        train_yvar.expand_as(train_obj),
        input_transform=Normalize(d=train_x.shape[-1]),
        outcome_transform=Standardize(m=1),
    ).to(train_x)
    model_con = SingleTaskGP(
        train_x,
        train_con,
        train_yvar.expand_as(train_con),
        input_transform=Normalize(d=train_x.shape[-1]),
        outcome_transform=Standardize(m=1),
    ).to(train_x)
    # combine into a multi-output GP model
    model = ModelListGP(model_obj, model_con)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model

EPSILON = 1e-6
def outcome_constraint(X):
    """constraint; The two elements of X are not zero."""
    constraint1 = EPSILON - torch.abs(X[..., 0])
    constraint2 = EPSILON - torch.abs(X[..., 1])
    return torch.max(constraint1, constraint2)

def obj_callable(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
    return Z[..., 0]

def constraint_callable(Z):
    return Z[..., 1]

objective = GenericMCObjective(objective=obj_callable)

def optimize_acqf_and_get_observation(acq_func):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    candidates, acq_value = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
    )
    # observe new values
    new_x = candidates.detach()
    new_con = outcome_constraint(new_x).unsqueeze(-1)  # add output dimension
    return new_x, new_con, acq_value

# =================以下Hops実装部分=================
# ==================Fase 1 ==================
app = Flask(__name__)
hops = hs.Hops(app)

@hops.component(
    "/get_points",
    name="Get Points",
    description="Get_Points\nGet points and reset points",
    inputs=[
        hs.HopsNumber("dim", "dim", "Dimention", hs.HopsParamAccess.ITEM),
        hs.HopsBoolean("Update", "Update", "Update Boolean(Toggle)"),
        hs.HopsBoolean("Reset", "Reset", "Reset Boolean(Toggle)"),
    ],
    outputs=[
        hs.HopsNumber("X", "X", access=hs.HopsParamAccess.TREE),
    ]
)
def get_points(dim: int,Update: bool,Reset: bool):
    if Reset:
        dim = int(dim)
        init_X = get_initial_points(dim, dim*2)
        torch.save(init_X, TENSOR_X_DATA_PATH)
    
    if not Update:
        return None
    tensor_X = torch.load(TENSOR_X_DATA_PATH)
    X_list = tensor_X.flatten().tolist()
    
    return X_list

@hops.component(
    "/single_model_loop",
    name="Single Task GP Model Loop",
    inputs=[
        hs.HopsNumber("Y", "Y", "評価変数",hs.HopsParamAccess.TREE),
        hs.HopsBoolean("Run", "R", "Trueにすると実行します"),
        hs.HopsBoolean("Create 0 img", "Create 0 img", "Reset Boolean(Toggle)"),
    ]
)
def single_model_loop(Y: float, run_execution: bool, Create: bool):
    train_x = torch.load(TENSOR_X_DATA_PATH)
    train_y = torch.tensor(list(Y.values()), dtype=dtype, device=device).view(-1, 1)

    if Create:
        dims_to_vary_list = list(itertools.combinations(range(6), 2))
        fig, axes = plt.subplots(5, 3, figsize=(18, 25))
        fig.suptitle('6次元データの散布図（全15パターンの2次元断面）', fontsize=22)
        
        train_x_np = train_x.cpu().numpy()
        train_y_np = train_y.cpu().numpy().flatten()

        for i, dims_to_vary in enumerate(dims_to_vary_list):
            ax = axes[i // 3, i % 3]
            dim1, dim2 = dims_to_vary
            scatter = ax.scatter(
                train_x_np[:, dim1], 
                train_x_np[:, dim2], 
                c=train_y_np, 
                cmap='viridis',
                alpha=0.7
            )
            ax.set_title(f'$X_{{{dim1+1}}}, X_{{{dim2+1}}}$ 平面', fontsize=16)
            ax.set_xlabel(f'$X_{{{dim1+1}}}$', fontsize=12)
            ax.set_ylabel(f'$X_{{{dim2+1}}}$', fontsize=12)
            ax.set_aspect('equal')
            fig.colorbar(scatter, ax=ax, label='Evaluation (y)')
            
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        plt.savefig(f'{IMG_PATH}\{0}.png')
        plt.close(fig)
        return
    
    if not run_execution:
        return

    train_con = outcome_constraint(train_x).unsqueeze(-1)
    mll, model = singletask_model(train_x, train_y, train_con)

    fit_gpytorch_mll(mll)

    MC_SAMPLES = 256
    qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

    qLogNEI = qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=train_x,
        sampler=qmc_sampler,
        objective=objective,
        constraints=[constraint_callable],
    )

    new_x, new_con, acq_value = optimize_acqf_and_get_observation(qLogNEI)

    new_train_x = torch.cat([train_x, new_x])
    new_train_con = torch.cat([train_con, new_con])

    torch.save(model, MODEL_PATH)
    torch.save(model.state_dict(), MODEL_OBJECT_PATH)
    torch.save(new_train_x, TENSOR_X_DATA_PATH)
    torch.save(new_train_con, TENSOR_CON_DATA_PATH)

    train_x_np = train_x.cpu().numpy()
    train_y_np = train_y.cpu().numpy().flatten()
    new_x_np = new_x.cpu().numpy()
    
    # 新しいデータポイントの予測値と標準偏差を計算
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = model.posterior(new_x)
        # Assuming the first output of the model is the objective for prediction
        mean = posterior.mean[..., 0].cpu().numpy().flatten()
        stddev = torch.sqrt(posterior.variance[..., 0]).cpu().numpy().flatten()

    dims_to_vary_list = list(itertools.combinations(range(6), 2))

    for i in range(len(new_x_np)):
        fig, axes = plt.subplots(5, 3, figsize=(18, 25))
        highlight_point_number = len(train_y_np) + i
        fig.suptitle(f'6次元データの散布図（全15パターン) #{highlight_point_number} 獲得関数:{acq_value:.2f}', fontsize=22)

        for j, dims_to_vary in enumerate(dims_to_vary_list):
            ax = axes[j // 3, j % 3]
            dim1, dim2 = dims_to_vary
            scatter = ax.scatter(
                train_x_np[:, dim1], 
                train_x_np[:, dim2], 
                c=train_y_np, 
                cmap='viridis',
                alpha=0.7
            )
            # 新しいデータをハイライト表示
            ax.scatter(
                new_x_np[i, dim1], 
                new_x_np[i, dim2], 
                color='red', 
                s=100, 
                edgecolors='white',
                linewidth=1.5,
                label='New point'
            )
            text_x = new_x_np[i, dim1]
            text_y = new_x_np[i, dim2]

            # 予測範囲（range）を計算
            lower_bound = (mean[i] - stddev[i])*100
            upper_bound = (mean[i] + stddev[i])*100
            # 標準偏差の値に基づいてカテゴリ名を決定
            if 0.10 < stddev[i] <= 0.15:
                category_name = '探索案'
            elif 0.05 < stddev[i] <= 0.10:
                category_name = '中間案'
            elif 0 <= stddev[i] <= 0.05:
                category_name = '活用案'
            # テキストを範囲表示に変更
            ax.text(
                text_x, 
                text_y, 
                f'{category_name}\nRange: {lower_bound:.0f} - {upper_bound:.0f}', 
                fontsize=15, 
                ha='left', 
                va='bottom', 
                color='black'
                #bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
            )
            ax.set_title(f'$X_{{{dim1+1}}}, X_{{{dim2+1}}}$ 平面', fontsize=16)
            ax.set_xlabel(f'$X_{{{dim1+1}}}$', fontsize=12)
            ax.set_ylabel(f'$X_{{{dim2+1}}}$', fontsize=12)
            ax.set_aspect('equal')
            fig.colorbar(scatter, ax=ax, label='Evaluation (y)')

        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        plt.savefig(f'{IMG_PATH}\{highlight_point_number}.png')
        plt.close(fig)


if __name__ == '__main__':
    app.run(debug=True)