import torch

# Taylor approximation for E[log(softplus(z))] and E[softplus(z)] 
def E_log_softplus_taylor(u, sigma2):  # Taylor series for E[log]
    """
    Taylor approximation for E[log(softplus(z))] where z ~ N(u, sigma2)
    Expansions done around the Taylor approximation at zero.
    Approximations are separated for different ranges of u.

    Parameters:
    -----------
    u : torch.Tensor
        Mean of the normal distribution
    sigma2 : torch.Tensor
        Variance of the normal distribution

    Returns:
    --------
    result : torch.Tensor
        Taylor approximation of E[log(softplus(z))]
    """
    result = torch.empty_like(u)

    # u < 2, taylor0 + log(exp(z))
    def weighted(u, sigma2):
        log2 = torch.log(torch.tensor(2.0, device=u.device))

        # Taylor expansion at zero
        taylor0 = (
            torch.log(log2)
            + (0.5 / log2) * u
            + ((log2 - 1) / (8 * log2**2)) * (u**2 + sigma2)
        )

        # Weighting term
        w = 1 / (1 + torch.exp(2 * (u + 2)))

        return (1 - w) * taylor0 + w * u

    # 2 <= u < 10, log(z + exp(-z))
    def approx(u, sigma2):
        return (
            torch.log(u + torch.exp(-u))
            - ((1 - torch.exp(-u)) ** 2) / (2 * (u + torch.exp(-u)) ** 2) * sigma2
        )

    # u >= 10, logz
    def log_approx(u, sigma2):
        return torch.log(u)  # - sigma2 / (2 * u**2)

    result[u < 2] = weighted(u[u < 2], sigma2[u < 2])
    result[(u >= 2) & (u < 10)] = approx(
        u[(u >= 2) & (u < 10)], sigma2[(u >= 2) & (u < 10)]
    )
    result[u >= 10] = log_approx(u[u >= 10], sigma2[u >= 10])

    return result


def E_softplus_taylor(u, sigma2):  # Taylor series for E[softplus]
    """
    Taylor approximation for E[softplus(z)] where z ~ N(u, sigma2)
    Expansions done around the Taylor approximation < 5
    Approximation is separated for different ranges of u.
    
    Parameters:
    -----------
    u : torch.Tensor
        Mean of the normal distribution
    sigma2 : torch.Tensor
        Variance of the normal distribution

    Returns:
    --------
    result : torch.Tensor
        Taylor approximation of E[softplus(z)]
    """
    result = torch.empty_like(u)

    # u < 5, taylor(u)
    def taylor(u, sigma2):
        softplus = torch.nn.functional.softplus(u)
        sig = torch.sigmoid(u)
        return softplus + 0.5 * sigma2 * sig * (1 - sig)

    # u >= 5, z #+ exp(-z)
    def approx(u, sigma2):
        return u  # + torch.exp(-u + sigma2/2)

    result[u < 5] = taylor(u[u < 5], sigma2[u < 5])
    result[u >= 5] = approx(u[u >= 5], sigma2[u >= 5])

    return result


# Monte Carlo approximation for E[log(softplus(z))] and E[softplus(z)] with reparameterization
def E_log_softplus_MC(u, sigma2, n_samples=10000):
    """
    Monte Carlo approximation for E[log(softplus(z))] where z ~ N(u, sigma2)
    Uses reparameterization trick: z = u + sqrt(sigma2) * epsilon, where epsilon ~ N(0,1)
    
    Parameters:
    -----------
    u : torch.Tensor
        Mean of the normal distribution
    sigma2 : torch.Tensor
        Variance of the normal distribution
    n_samples : int
        Number of Monte Carlo samples
        
    Returns:
    --------
    result : torch.Tensor
        Monte Carlo approximation of E[log(softplus(z))]
    """
    # Reparameterization: z = u + sqrt(sigma2) * epsilon, epsilon ~ N(0,1)
    epsilon = torch.randn(n_samples, *u.shape, device=u.device, dtype=u.dtype) # (n_samples, ...)
    z = u.unsqueeze(0) + torch.sqrt(sigma2).unsqueeze(0) * epsilon
    
    # Apply log(softplus) transformation
    log_softplus_z = torch.log(torch.nn.functional.softplus(z))
    
    # Take the mean over samples dimension
    result = torch.mean(log_softplus_z, dim=0)
    
    return result


def E_softplus_MC(u, sigma2, n_samples=10000):
    """
    Monte Carlo approximation for E[softplus(z)] where z ~ N(u, sigma2)
    Uses reparameterization trick: z = u + sqrt(sigma2) * epsilon, where epsilon ~ N(0,1)
    
    Parameters:
    -----------
    u : torch.Tensor
        Mean of the normal distribution
    sigma2 : torch.Tensor
        Variance of the normal distribution
    n_samples : int
        Number of Monte Carlo samples
        
    Returns:
    --------
    result : torch.Tensor
        Monte Carlo approximation of E[softplus(z)]
    """
    # Reparameterization: z = u + sqrt(sigma2) * epsilon, epsilon ~ N(0,1)
    epsilon = torch.randn(n_samples, *u.shape, device=u.device, dtype=u.dtype)
    z = u.unsqueeze(0) + torch.sqrt(sigma2).unsqueeze(0) * epsilon
    
    # Apply softplus transformation
    softplus_z = torch.nn.functional.softplus(z)
    
    # Take the mean over samples dimension
    result = torch.mean(softplus_z, dim=0)
    
    return result


def E_log_exp(u, sigma2):
    """
    Approximation for E[ln(exp(z))] where z ~ N(u, sigma2)
    """
    return u

def E_exp(u, sigma2):
    """
    Approximation for E[exp(z)] where z ~ N(u, sigma2)
    """
    return torch.exp(u + sigma2 / 2)

    
def E_log_r_softplus_MC(u, sigma2, r, lib, n_samples=10000):
    """
    Monte Carlo approximation for E[log(r + softplus(z))] where z ~ N(u, sigma2)
    Uses reparameterization trick: z = u + sqrt(sigma2) * epsilon, where epsilon ~ N(0,1)
    
    Parameters:
    -----------
    u : torch.Tensor
        Mean of the normal distribution
    sigma2 : torch.Tensor
        Variance of the normal distribution
    r : float
        Dispersion parameter
    lib : float
        Library size factor
    n_samples : int
        Number of Monte Carlo samples
        
    Returns:
    --------
    result : torch.Tensor
        Monte Carlo approximation of E[log(r + softplus(z))]
    """
    # Reparameterization: z = u + sqrt(sigma2) * epsilon, epsilon ~ N(0,1)
    epsilon = torch.randn(n_samples, *u.shape, device=u.device, dtype=u.dtype)
    z = u.unsqueeze(0) + torch.sqrt(sigma2).unsqueeze(0) * epsilon
    
    # Apply log(r + softplus) transformation
    log_r_softplus_z = torch.log(r + lib * torch.nn.functional.softplus(z))
    
    # Take the mean over samples dimension
    result = torch.mean(log_r_softplus_z, dim=0)
    
    return result


def E_log_r_exp(u, sigma2, log_r, lib, n_samples=10000):
    """
    Monte Carlo approximation for E[log(r + exp(z))] where z ~ N(u, sigma2)
    Uses reparameterization trick: z = u + sqrt(sigma2) * epsilon, where epsilon ~ N(0,1)
    
    Parameters:
    -----------
    u : torch.Tensor
        Mean of the normal distribution
    sigma2 : torch.Tensor
        Variance of the normal distribution
    r : float
        Dispersion parameter
    lib : float
        Library size factor
    n_samples : int
        Number of Monte Carlo samples
        
    Returns:
    --------
    result : torch.Tensor
        Monte Carlo approximation of E[log(r + exp(z))]
    """
    # Reparameterization: z = u + sqrt(sigma2) * epsilon, epsilon ~ N(0,1)
    epsilon = torch.randn(n_samples, *u.shape, device=u.device, dtype=u.dtype)
    z = u.unsqueeze(0) + torch.sqrt(sigma2).unsqueeze(0) * epsilon
    
    # Apply log(r + exp) transformation
    log_r_exp_z = torch.logaddexp(log_r, torch.log(lib) + z)
    
    # Take the mean over samples dimension
    result = torch.mean(log_r_exp_z, dim=0)
    
    return result

