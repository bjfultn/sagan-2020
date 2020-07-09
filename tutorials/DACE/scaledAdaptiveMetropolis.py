# -*- coding: utf-8 -*-

# Copyright 2019 Jean-Baptiste Delisle

import numpy as np

def sam(
  x0, logprob, nsamples=100000,
  cov0=None,
  cov_update_interval=100,
  cov_update_law=lambda t: (100/(t+1))**(2/3),
  scale0=None,
  accept_rate_target=0.234,
  print_level=1, print_interval=1000, print_inplace=True,
  **kwargs):
  """
  Adaptive Metropolis algorithm (Haario et al. 2001)
  with adaptive scaling (Andrieu & Thoms 2008).

  Parameters
  ----------
  x0 : (ndim,) ndarray
    Initial guess of the parameters.
  logprob(x, **kwargs) : function
    Log probability of the distribution to sample.
  nsamples: int
    Number of samples to draw.
  cov0 :  (ndim, ndim) ndarray
    Initial guess of the parameters' covariance matrix.
  cov_update_interval : int
    Interval at which to update the covariance matrix.
  cov_update_law(t) : function
    Update coefficient (between 0 and 1)
    as a function of time (should vanish as t -> nsamples)
  scale0 : float
    Initial scaling factor for the proposal distribution
    (by default 2.4/sqrt(ndim)).
  accept_rate_target : float
    Acceptance rate to target when rescaling the proposal distribution,
    if None, the scale is left unchanged.
  print_level : int
    0 (no printing)
    1 (print acceptance rate)
  print_interval : int
    Interval at which to print infos.
  print_inplace : bool
    Whether to print infos in place or one line after the other.
  **kwargs :
    Additional parameters for the logprob function.

  Returns
  -------
  samples : (nsamples+1, ndim) ndarray
    Array of parameters values for each sample.
  diagnositics : dict
    Dictionary of diagnostics, with the following keys:
    logprob : (nsamples+1,) ndarray
      Array of log probability for each sample.
    alpha : (nsamples,) ndarray
      Array of acceptance probability for each proposal.
    accept : (nsamples,) ndarray
      Array of acceptance for each proposal.
    mu : (ndim,) ndarray
      Final estimate of the mean.
    cov : (ndim, ndim) ndarray
      Final estimate of the covariance matrix.
  """

  nsamples = int(nsamples)
  ndim = len(x0)
  # Init state
  x = x0
  lpx = logprob(x, **kwargs)
  # Init chain
  histx = np.empty((nsamples+1, ndim))
  histlpx = np.empty(nsamples+1)
  histalpha = np.empty(nsamples)
  histaccept = np.empty(nsamples, dtype=bool)
  histx[0] = x
  histlpx[0] = lpx
  # Init covariance matrix
  mu = x0
  if scale0 is None:
    scale = 2.4/np.sqrt(ndim)
  else:
    scale = scale0
  if cov0 is None:
    C = np.identity(ndim)
  else:
    if cov0.shape != (ndim,ndim):
      raise Exception('Incompatible shapes for x0 ({:d}) and cov0 {:s}.'
        .format(ndim, str(cov0.shape)))
    C = cov0
  # SVD decomposition of C (more robust than Cholesky)
  _, s, v = np.linalg.svd(C)
  sqCT = np.sqrt(s)[:, None] * v
  use_empirical_cov = False
  # Init printing
  print_fmt = (
    '{}Step {{:{:d}d}}, acceptance rate (since last printing): {{:.4f}}'
    .format('\r' if print_inplace else '', 1+int(np.log10(nsamples))))
  print_end = ' ' if print_inplace else None

  # Big loop
  for t in range(1,nsamples+1):
    # Proposal of new point (y)
    y = x + np.random.normal(scale=scale, size=ndim).dot(sqCT)
    # Compute proposal probability
    lpy = logprob(y, **kwargs)
    # Do we accept the proposal
    alpha = np.exp(min(0.0,lpy-lpx))
    accept = np.random.random() < alpha
    if accept:
      x = y
      lpx = lpy
    # Save state in chain
    histx[t] = x
    histlpx[t] = lpx
    histalpha[t-1] = alpha
    histaccept[t-1] = accept
    # Update covariance matrix
    if t%cov_update_interval==0:
      gamma = cov_update_law(t)
      # Adapt scale
      if accept_rate_target is not None and use_empirical_cov:
        mean_alpha = np.mean(histalpha[t-cov_update_interval:t])
        scale *= ((mean_alpha+0.25*accept_rate_target)/
          (1.25*accept_rate_target))**gamma
      # Update mean
      mudt = np.mean(histx[t+1-cov_update_interval:t+1], axis=0)
      dmu = mudt - mu
      mu += gamma*dmu
      # Update cov
      Cdt = np.cov(histx[t+1-cov_update_interval:t+1], rowvar=False)
      dmu.shape = (ndim,1)
      C = (1.0-gamma)*C + gamma*Cdt + gamma*(1.0-gamma)*dmu.dot(dmu.T)
      if not use_empirical_cov:
        mean_accept = np.mean(histaccept[t-cov_update_interval:t])
        if mean_accept > 0.1:
          use_empirical_cov = True
        else:
          coef = (mean_accept+0.025)/0.125
          C *= coef*coef
          sqCT *= coef
      if use_empirical_cov:
        # SVD decomposition of C (more robust than Cholesky)
        _, s, v = np.linalg.svd(C)
        sqCT = np.sqrt(s)[:, None] * v
    # Print infos
    if print_level and t%print_interval == 0:
      print(print_fmt.format(t, np.mean(histaccept[t-print_interval:t])),
        end=print_end)
  if print_level and print_inplace and nsamples>=print_interval:
    print()
  return(
    histx,
    {
      'logprob': histlpx,
      'alpha': histalpha,
      'accept': histaccept,
      'mu': mu,
      'cov': C
    }
  )
