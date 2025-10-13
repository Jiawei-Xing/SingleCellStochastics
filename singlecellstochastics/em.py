from .optimize import Lq_optimize_torch

def run_em(
    init_params,
    mode,
    x_tensor_list,
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
):
    """
    Run EM algorithm for Lq optimization.

    init_params: (batch_size, N_sim, all_param_dim)
    mode: 1 for H0, 2 for H1
    x_original_tensor: list of (batch_size, N_sim, n_regimes)
    gene_names: list of gene names
    diverge_list_torch: list of (batch_size, N_sim, n_regimes)
    share_list_torch: list of (batch_size, N_sim, n_regimes)
    epochs_list_torch: list of (batch_size, N_sim, n_regimes)
    beta_list_torch: list of (batch_size, N_sim, n_regimes)
    max_iter: int
    learning_rate: float
    device: str
    wandb_flag: bool
    window: int
    tol: float
    approx: str
    em_iter: int
    prior: float
    kkt: bool
    Returns: (batch_size, N_sim, all_param_dim), (batch_size, N_sim)
    """
    for i in range(em_iter):
        # E-step
        h0_params, h0_loss = Lq_optimize_torch(
            init_params,
            mode,
            x_tensor_list,
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
            "e",
            prior,
            kkt
        )  # (batch_size, N_sim, all_param_dim)

        # M-step
        h0_params, h0_loss = Lq_optimize_torch(
            h0_params,
            mode,
            x_tensor_list,
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
            "m",
            prior,
            kkt
        )  # (batch_size, N_sim, all_param_dim)

        # update init_params for next iteration
        init_params = h0_params

    # run last E-step
    h0_params, h0_loss = Lq_optimize_torch(
        init_params,
        mode,
        x_tensor_list,
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
        "e",
        prior,
        kkt
    )  # (batch_size, N_sim, all_param_dim)

    h0_params = [p.clone().detach() for p in h0_params]
    return h0_params, h0_loss.clone().detach()
