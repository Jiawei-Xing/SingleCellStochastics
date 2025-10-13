import numpy as np
import torch
import matplotlib.pyplot as plt

from .optimize import ou_optimize_scipy, ou_optimize_torch, Lq_optimize_torch
from .em import run_em

# likelihood ratio test
def likelihood_ratio_test(
    expr,
    n_regimes,
    diverge_list,
    share_list,
    epochs_list,
    beta_list,
    diverge_list_torch,
    share_list_torch,
    epochs_list_torch,
    beta_list_torch,
    gene_names,
    max_iter,
    learning_rate,
    device,
    wandb_flag,
    window,
    tol,
    approx, 
    em_iter,
    pseudo,
    batch_start,
    prior,
    init,
    kkt
):
    """
    Hypothesis testing for lineage-specific gene expression change.

    expr: (batch_size, N_sim, n_cells) numpy array
    diverge_list, share_list, epochs_list, beta_list: list of numpy arrays
    diverge_list_torch, share_list_torch, epochs_list_torch, beta_list_torch: list of tensors
    gene_names: list of gene names
    window: number of iterations to check convergence
    tol: convergence tolerance
    approx: approximation method
    em_iter: number of EM iterations
    pseudo: pseudo count for reverse softplus read counts as inital mean
    batch_start: start index of the batch
    prior: prior for alpha
    init: whether to initialize with OU optimization
    kkt: whether to use KKT condition in optimization
    Returns: (batch_size, N_sim) numpy array of params and losses
    """
    # Initialize OU means with expression data
    if pseudo > 0:
        x_pseudo = [
            np.maximum(x, pseudo) for x in expr
        ]  # add small value to avoid log(0)
        m_init = [
            np.log(np.expm1(x)) if approx != "exp"
            else np.log(x)
            for x in x_pseudo
        ] # reverse read counts through softplus or exp
    else:
        m_init = [
            np.log1p(x) for x in expr # log normalize of read counts
        ]

    # Initialize q standard deviation with expression data
    s_init = [
        np.tile(
            np.maximum(1e-6, np.std(x, axis=-1, keepdims=True)), 
            (1, 1, x.shape[-1])
        ) for x in m_init
    ]

    '''
    # use the first gene as the example gene
    m_first = [
        m[0:1, 0:1, :] for m in m_init
    ] # (n_cells,)

    # init OU parameters with one example gene
    ou_params_init = np.ones((n_regimes + 2))  # shape: (n_regimes+2)
    ou_params_init_h0 = ou_optimize_scipy(
        ou_params_init,
        1,
        m_first,
        diverge_list,
        share_list,
        epochs_list,
        beta_list,
        device=device,
    )  # (B, S, n_regimes+2)
    '''

    # Initialize OU means for torch
    m_init_tensor = [
        torch.tensor(m, dtype=torch.float32, device=device) for m in m_init
    ]  # list of (batch_size, N_sim, n_cells)
    
    # Initialize OU parameters
    batch_size, N_sim, _ = m_init_tensor[0].shape
    ou_params_init = torch.ones(
        (batch_size, N_sim, n_regimes + 2), dtype=torch.float32, device=device
    )

    # Initial run with OU model
    if init:
        ou_params_h0, ou_loss_h0 = ou_optimize_torch(
            ou_params_init,
            2,
            m_init_tensor,
            diverge_list_torch,
            share_list_torch,
            epochs_list_torch,
            beta_list_torch,
            device,
            max_iter,
            learning_rate,
            wandb_flag,
            gene_names,
            window,
            tol,
            kkt
        )
        ou_params_init = ou_params_h0

    # Initialize q std for torch
    s_init_tensor = [
        torch.tensor(s, dtype=torch.float32, device=device) for s in s_init
    ]

    # Initialize all Lq parameters for torch
    pois_params_init = [
        torch.cat((m_init_tensor[i], s_init_tensor[i]), dim=-1) for i in range(len(m_init))
    ]  # list of (batch_size, N_sim, 2*n_cells)

    init_params = pois_params_init + [ou_params_init]  # list of (batch_size, N_sim, param_dim)
 
    # Convert expression data to torch tensor
    x_tensor = [
        torch.tensor(x, dtype=torch.float32, device=device) for x in expr
    ]  # list of (batch_size, N_sim, n_cells)

    # optimize Lq for null model
    if em_iter == 0: # optimize all params
        h0_params, h0_loss = Lq_optimize_torch(
            init_params,
            1,
            x_tensor,
            gene_names,
            diverge_list_torch,
            share_list_torch,
            epochs_list_torch,
            beta_list_torch,
            max_iter,
            learning_rate,
            device,
            wandb_flag,
            window,
            tol,
            approx,
            None,
            prior,
            kkt
        )  # (batch_size, N_sim, all_param_dim), (batch_size, N_sim)
    else: # EM
        h0_params, h0_loss = run_em(
            init_params,
            1,
            x_tensor,
            gene_names,
            diverge_list_torch,
            share_list_torch,
            epochs_list_torch,
            beta_list_torch,
            max_iter,
            learning_rate,
            device,
            wandb_flag,
            window,
            tol,
            approx,
            em_iter,
            prior,
            kkt
        )  # (batch_size, N_sim, all_param_dim), (batch_size, N_sim)
    
    alphas = [i+1 for i in range(200)]
    elbos = []
    for alpha in alphas: # try different initial alpha
        init_params[-1][0, 0, 0] = np.log(alpha)
        h0_params_grid, h0_loss_grid = Lq_optimize_torch(
            init_params,
            1,
            x_tensor,
            gene_names,
            diverge_list_torch,
            share_list_torch,
            epochs_list_torch,
            beta_list_torch,
            1,
            learning_rate,
            device,
            wandb_flag,
            window,
            tol,
            approx,
            None,
            prior,
            kkt
        )  # (batch_size, N_sim, all_param_dim), (batch_size, N_sim)
        elbos.append(h0_loss_grid.clone().detach().cpu().numpy()[0, 0])

    plt.plot(alphas, elbos, marker='o', label='-ELBO', ms=2)
    plt.xlabel("alpha")
    plt.ylabel("-ELBO")
    plt.title("h0 ELBO")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    best_idx = int(np.argmin(elbos))
    plt.scatter(alphas[best_idx], elbos[best_idx], color="red", zorder=5, label="elbo best α")
    plt.legend()
    plt.savefig(f"{batch_start}_h0_elbo.png")
    plt.show()
    plt.close()
    print(f"h0 best alpha: {alphas[int(np.argmin(elbos))]}, elbo: {np.min(elbos)}")

    # optimize Lq for alternative model
    if em_iter == 0: # optimize all params
        h1_params, h1_loss = Lq_optimize_torch(
            h0_params,
            2,
            x_tensor,
            gene_names,
            diverge_list_torch,
            share_list_torch,
            epochs_list_torch,
            beta_list_torch,
            max_iter,
            learning_rate,
            device,
            wandb_flag,
            window,
            tol,
            approx,
            None,
            prior,
            kkt
        )  # (batch_size, N_sim, all_param_dim), (batch_size, N_sim)
    else: # EM
        h1_params, h1_loss = run_em(
            h0_params,
            2,
            x_tensor,
            gene_names,
            diverge_list_torch,
            share_list_torch,
            epochs_list_torch,
            beta_list_torch,
            max_iter,
            learning_rate,
            device,
            wandb_flag,
            window,
            tol,
            approx,
            em_iter,
            prior,
            kkt
        )  # (batch_size, N_sim, all_param_dim), (batch_size, N_sim)
    
    elbos = []
    for alpha in alphas: # try different initial alpha
        init_params[-1][0, 0, 0] = np.log(alpha)
        h1_params_grid, h1_loss_grid = Lq_optimize_torch(
            init_params,
            2,
            x_tensor,
            gene_names,
            diverge_list_torch,
            share_list_torch,
            epochs_list_torch,
            beta_list_torch,
            1,
            learning_rate,
            device,
            wandb_flag,
            window,
            tol,
            approx,
            None,
            prior,
            kkt
        )  # (batch_size, N_sim, all_param_dim), (batch_size, N_sim)
        elbos.append(h1_loss_grid.clone().detach().cpu().numpy()[0, 0])

    plt.plot(alphas, elbos, marker='o', label='-ELBO', ms=2)
    plt.xlabel("alpha")
    plt.ylabel("-ELBO")
    plt.title("h1 ELBO")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    best_idx = int(np.argmin(elbos))
    plt.scatter(alphas[best_idx], elbos[best_idx], color="red", zorder=5, label="elbo best α")
    plt.legend()
    plt.savefig(f"{batch_start}_h1_elbo.png")
    plt.show()
    plt.close()
    print(f"h1 best alpha: {alphas[int(np.argmin(elbos))]}, elbo: {np.min(elbos)}")
    
    return h0_params[-1].clone().detach().cpu().numpy(), \
        h0_loss.clone().detach().cpu().numpy(), \
        h1_params[-1].clone().detach().cpu().numpy(), \
        h1_loss.clone().detach().cpu().numpy()

    