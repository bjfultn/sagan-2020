# -*- coding: utf-8 -*-

# Copyright 2019 Jean-Baptiste Delisle

import numpy as np
from scipy.optimize import minimize
from spleaf.rv import Cov
from kepderiv import Keplerian
import tools
from scaledAdaptiveMetropolis import sam

default_inst_sec_acc = {
	'CORALIE98': True,
	'CORALIE07': True,
	'CORALIE14': True,
	'CORAVEL': True,
	'HARPS03': True,
	'HARPS15': True,
	'HARPN': True,
	'HIRES': False,
	'UCLES': False,
	'ESPRESSO': True,
	'SOPHIE': True,
	'ELODIE': True,
	'default': False
}

default_inst_jitter = {
	'CORALIE98': 5.0,
	'CORALIE07': 8.0,
	'CORALIE14': 3.0,
	'CORAVEL': 150.0,
	'HARPS03': 0.75,
	'HARPS15': 0.75,
	'HARPN': 0.75,
	'HIRES': 2.5,
	'UCLES': 7.0,
	'ESPRESSO': 0.05,
	'SOPHIE': 1.5,
	'ELODIE': 12.0,
	'default': 0.0
}

class rvModel():
  def __init__(self, t, rv, *args, **kwargs):
    self.rv = rv
    self.cov = Cov(t, *args, **kwargs)
    self.t = t
    self.n = self.cov.n

    self.npla = 0
    self.plauid = 0
    self.planame = []
    self.kep = []

    self.nlin = 0
    self.linuid = 0
    self.linname = []
    self.linpar = np.empty(0)
    self.linM = np.empty((0,self.n))

    self.fitparams = []

  def addlin(self, derivative, name=None, value=0.0, fitted=True):
    self.linM = np.vstack((self.linM, derivative))
    if name is None:
      name = '{}'.format(self.linuid)
    self.linname.append(name)
    self.linpar = np.concatenate((self.linpar,[value]))
    if fitted:
      self.fitparams.append('lin.{}'.format(name))
    self.nlin += 1
    self.linuid += 1

  def rmlin(self, name):
    klin = self.linname.index(name)
    self.linM = np.delete(self.linM, klin, 0)
    self.linname.pop(klin)
    self.linpar = np.delete(self.linpar, klin, 0)
    paramname = 'lin.{}'.format(name)
    if paramname in self.fitparams:
      self.fitparams.remove(paramname)
    self.nlin -= 1

  def addpla(self, values, name=None, fitted=True, params=['n', 'M0', 'K', 'e', 'omega']):
    if name is None:
      name = '{}'.format(self.plauid)
    self.planame.append(name)
    self.kep.append(Keplerian(values, params))
    if fitted:
      for par in params:
        self.fitparams.append('pla.{}.{}'.format(name,par))
    self.npla += 1
    self.plauid += 1

  def smartaddpla(self, P, name=None, fitted=True):
    res = self.residuals()
    u = self.cov.solveL(res)/self.cov.sqD()
    nu = 2*np.pi/P
    nut_rad = nu*self.t
    Mt = np.concatenate(([np.cos(nut_rad), np.sin(nut_rad), np.cos(2*nut_rad), np.sin(2*nut_rad)],
      self.linM))
    Nt = np.array([self.cov.solveL(Mk)/self.cov.sqD() for Mk in Mt])
    x = np.linalg.inv(Nt@Nt.T)@Nt@u
    e, M0_rad = tools.calc_eM(x[:4])
    Mt = np.concatenate((tools.designMatrix_Kom(self.t, P, e, M0_rad), self.linM))
    Nt = np.array([self.cov.solveL(Mk)/self.cov.sqD() for Mk in Mt])
    x = np.linalg.inv(Nt@Nt.T)@Nt@u
    K = np.sqrt(x[0]**2+x[1]**2)
    omega_rad = np.arctan2(x[1], x[0])
    self.addpla(np.array([nu, M0_rad, K, e, omega_rad]), name, fitted)

  def changeparpla(self, name, params=['n', 'M0', 'K', 'e', 'omega']):
    kpla = self.planame.index(name)
    fitted = False
    for par in reversed(self.fitparams):
      if par.startswith('pla.{}.'.format(name)):
        self.fitparams.remove(par)
        fitted = True
    self.kep[kpla].set_param(params)
    if fitted:
      for par in params:
        self.fitparams.append('pla.{}.{}'.format(name,par))

  def rmpla(self, name):
    kpla = self.planame.index(name)
    self.planame.pop(kpla)
    self.kep.pop(kpla)
    for par in reversed(self.fitparams):
      if par.startswith('pla.{}.'.format(name)):
        self.fitparams.remove(par)
    self.npla -= 1

  def model(self, t=None, x=None, paramlist=None, backup=True):
    if t is None:
      t = self.t
    if x is not None:
      if backup:
        x_old = self.get_params(paramlist)
      self.set_params(x, paramlist)
    v = self.linpar.dot(self.linM)
    for pla in self.kep:
      v += pla.rv(t)
    if x is not None and backup:
      self.set_params(x_old, paramlist)
    return(v)

  def residuals(self, x=None, paramlist=None, backup=True):
    return(self.rv-self.model(x=x, paramlist=paramlist, backup=backup))

  def periodogram(self, nu0, dnu, nfreq):
    res = self.residuals()
    N0t = np.array([self.cov.solveL(M0k)/self.cov.sqD() for M0k in self.linM])
    u = self.cov.solveL(res)/self.cov.sqD()
    u2 = np.sum(u*u)
    N0tu = N0t@u
    chi20 = u2 - N0tu.T@np.linalg.inv(N0t@N0t.T)@N0tu

    nu = nu0 + np.arange(nfreq)*dnu
    chi2 = np.empty(nfreq)
    dnut_rad = dnu*self.t
    cosdnut = np.cos(dnut_rad)
    sindnut = np.sin(dnut_rad)
    nu0t_rad = nu0*self.t
    cosnut = np.cos(nu0t_rad)
    sinnut = np.sin(nu0t_rad)
    Nt = np.vstack(([self.cov.solveL(cosnut)/self.cov.sqD(),
      self.cov.solveL(sinnut)/self.cov.sqD()], N0t))
    Ntu = Nt@u
    chi2[0] = u2 - Ntu.T@np.linalg.inv(Nt@Nt.T)@Ntu
    for kfreq in range(1, nfreq):
      cosnut, sinnut = cosnut*cosdnut-sinnut*sindnut, sinnut*cosdnut+cosnut*sindnut
      Nt[0] = self.cov.solveL(cosnut)/self.cov.sqD()
      Nt[1] = self.cov.solveL(sinnut)/self.cov.sqD()
      Ntu[0] = Nt[0]@u
      Ntu[1] = Nt[1]@u
      chi2[kfreq] = u2 - Ntu.T@np.linalg.inv(Nt@Nt.T)@Ntu
    power = 1.0 - chi2/chi20
    return(nu, power)

  def psdw(self, nu0, dnu, nfreq):
    res = np.ones(self.n)
    u = self.cov.solveL(res)/self.cov.sqD()
    u2 = np.sum(u*u)
    chi20 = u2

    nu = nu0 + np.arange(nfreq)*dnu
    chi2 = np.empty(nfreq)
    dnut_rad = dnu*self.t
    cosdnut = np.cos(dnut_rad)
    sindnut = np.sin(dnut_rad)
    nu0t_rad = nu0*self.t
    cosnut = np.cos(nu0t_rad)
    sinnut = np.sin(nu0t_rad)
    Nt = np.array([self.cov.solveL(cosnut)/self.cov.sqD(),
      self.cov.solveL(sinnut)/self.cov.sqD()])
    Ntu = Nt@u
    chi2[0] = u2 - Ntu.T@np.linalg.inv(Nt@Nt.T)@Ntu
    for kfreq in range(1, nfreq):
      cosnut, sinnut = cosnut*cosdnut-sinnut*sindnut, sinnut*cosdnut+cosnut*sindnut
      Nt[0] = self.cov.solveL(cosnut)/self.cov.sqD()
      Nt[1] = self.cov.solveL(sinnut)/self.cov.sqD()
      Ntu[0] = Nt[0]@u
      Ntu[1] = Nt[1]@u
      chi2[kfreq] = u2 - Ntu.T@np.linalg.inv(Nt@Nt.T)@Ntu
    power = 1.0 - chi2/chi20
    return(nu, power)

  def Teff(self, numax):
    W = self.cov.expandInv()
    sinc = np.array([np.sinc(numax*(self.t-ti)) for ti in self.t])
    Wsinc = W*sinc
    Wsinct = Wsinc@self.t
    q = np.sum(Wsinc)
    s = np.sum(Wsinct)
    r = self.t@Wsinct
    return(2.0*np.sqrt(np.pi*(r/q-(s/q)**2)))

  def Teff_low(self, numax):
    su = self.cov.solveL(np.ones(self.cov.n))/self.cov.sqD()
    st = self.cov.solveL(self.t)/self.cov.sqD()
    q = np.sum(su*su)
    s = np.sum(su*st)
    r = np.sum(st*st)
    return(2.0*np.sqrt(np.pi*(r/q-(s/q)**2)))

  def fap(self, zmax, numax, Teff=None, Teff_approx=None):
    Nh = self.n - self.linM.shape[0]
    Nk = Nh - 2
    fmax = numax/(2.0*np.pi)
    if Teff is None:
      if Teff_approx == 'low':
        W = fmax * self.Teff_low(numax)
      else:
        W = fmax * self.Teff(numax)
    else:
      W = fmax * Teff
    chi2ratio = 1.0 - zmax
    FapSingle = chi2ratio**(Nk/2.0)
    tau = W * FapSingle * np.sqrt(Nh*zmax/(2.0*chi2ratio))
    Fap = FapSingle + tau
    if Fap > 1e-5:
      Fap = 1.0 - (1.0 - FapSingle) * np.exp(-tau)
    return(Fap)

  def chi2(self, x=None, paramlist=None, backup=True):
    if x is not None and backup:
      x_old = self.get_params(paramlist)
    chi2 = self.cov.chi2(self.residuals(x, paramlist, False))
    if x is not None and backup:
      self.set_params(x_old, paramlist)
    return(chi2)

  def loglike(self, x=None, paramlist=None, backup=True):
    if x is not None and backup:
      x_old = self.get_params(paramlist)
    ll = self.cov.loglike(self.residuals(x, paramlist, False))
    if x is not None and backup:
      self.set_params(x_old, paramlist)
    return(ll)

  def _func_jac(self, func_name, x=None, paramlist=None, backup=True):
    if x is not None and backup:
      x_old = self.get_params(paramlist)
    setattr(self, func_name+'_value', getattr(self, func_name)(x, paramlist, False))
    ( grad_res,
      grad_var_jitter,
      grad_var_jitter_inst, grad_var_calib_inst,
      grad_var_exp, grad_lambda_exp,
      grad_var_cos_qper, grad_var_sin_qper,
      grad_lambda_qper, grad_nu_qper ) = getattr(self.cov, func_name+'_grad')()
    grad_noise = {
      'var_jitter': grad_var_jitter,
      'var_jitter_inst': grad_var_jitter_inst,
      'var_calib_inst': grad_var_calib_inst,
      'var_exp': grad_var_exp,
      'lambda_exp': grad_lambda_exp,
      'var_cos_qper': grad_var_cos_qper,
      'var_sin_qper': grad_var_sin_qper,
      'lambda_qper': grad_lambda_qper,
      'nu_qper': grad_nu_qper
    }
    grad_pla = []
    for pla in self.kep:
      grad_pla.append(-pla.rv_back(grad_res))
    jac = []
    for paramname in self.fitparams:
      paramsplit = paramname.split('.')
      if paramsplit[0] == 'lin':
        jac.append(-self.linM[self.linname.index(paramsplit[1])].dot(grad_res))
      elif paramsplit[0] == 'pla':
        kpla = self.planame.index(paramsplit[1])
        kpar = self.kep[kpla].get_param().index(paramsplit[2])
        jac.append(grad_pla[kpla][kpar])
      elif paramsplit[0] == 'cov':
        if len(paramsplit) > 2:
          jac.append(grad_noise[paramsplit[1]][int(paramsplit[2])])
        else:
          jac.append(grad_noise[paramsplit[1]])
      else:
        raise Exception('unknown paramname ({}).'.format(paramname))
    if x is not None and backup:
      self.set_params(x_old, paramlist)
    return(np.array(jac))

  def chi2_jac(self, x=None, paramlist=None, backup=True):
    return(self._func_jac('chi2', x, paramlist, backup))

  def loglike_jac(self, x=None, paramlist=None, backup=True):
    return(self._func_jac('loglike', x, paramlist, backup))

  def loglike_hess(self, x=None, paramlist=None, backup=True, step=1e-6):
    if x is not None and backup:
      x_old = self.get_params(paramlist)
    llj0 = self.loglike_jac(x, paramlist, False)
    x0 = self.get_params(paramlist)
    xb = list(x0)
    nparam = len(x0)
    hess = np.empty((nparam,nparam))
    for k in range(nparam):
      xb[k] += step
      hess[k] = (self.loglike_jac(xb, paramlist, False) - llj0)/step
      xb[k] = x0[k]
    if x is not None and backup:
      self.set_params(x_old, paramlist)
    else:
      self.set_params(x0, paramlist)
    return((hess + hess.T)/2)

  def fit(self, method='L-BFGS-B', options=None, bounds='default', step_hess=1e-6):
    scale = 1/np.sqrt(np.abs(np.diag(self.loglike_hess(step=step_hess))))
    if bounds == 'default':
      bounds = []
      for paramname, sc in zip(self.fitparams, scale):
        # eccentricity:
        if paramname.startswith('pla.') and paramname.endswith('.e'):
          bounds.append((0, 0.95/sc))
        elif paramname.startswith('pla.') and (paramname.endswith('.k') or paramname.endswith('.h')
          or paramname.endswith('.sqk') or paramname.endswith('.sqh') or 'cos' in paramname or 'sin' in paramname):
          bounds.append((-0.95/sc, 0.95/sc))
        # variance
        elif paramname.startswith('cov.var') or (paramname.startswith('pla.') and paramname.endswith('.K')):
          bounds.append((0,None))
        else:
          bounds.append((None,None))
    x_old = self.get_params()
    result = minimize(lambda x:-self.loglike(x*scale, backup=False),
                      np.array(x_old)/scale,
                      jac=lambda x:-self.loglike_jac(x*scale, backup=False)*scale,
                      method=method,
                      options=options,
                      bounds=bounds)
    if result.success:
      self.set_params(result.x*scale)
    else:
      print(result)
      print()
      self.set_params(x_old)
      raise Exception('Fit did not converge.')

  def _get_dec_from_err(self, err):
    if err==0 or np.isnan(err):
      return(6)
    else:
      return(int(max(0, 3-np.log10(err))))

  def show_params(self, paramlist=None, step_hess=1e-6):
    if paramlist is None:
      paramlist = self.fitparams
    values = self.get_params(paramlist)
    hess = self.loglike_hess(paramlist=paramlist, step=step_hess)
    errors = np.sqrt(-np.diag(np.linalg.inv(hess)))
    print('{:25s} {:>12s}     {:12s}'.format('Parameter', 'Value', 'Error'))
    for paramname, value, err in zip(paramlist, values, errors):
      dec = self._get_dec_from_err(err)
      fmt = f'{{:25s}} {{:12.{dec}f}}  ±  {{:<12.{dec}f}}'
      print(fmt.format(paramname, value, err))
    print()

  def get_params(self, paramlist=None):
    singlevar = False
    if paramlist is None:
      paramlist = self.fitparams
    elif isinstance(paramlist, str):
      singlevar = True
      paramlist = [paramlist]
    values = []
    for paramname in paramlist:
      paramsplit = paramname.split('.')
      if paramsplit[0] == 'lin':
        values.append(self.linpar[self.linname.index(paramsplit[1])])
      elif paramsplit[0] == 'pla':
        kpla = self.planame.index(paramsplit[1])
        kpar = self.kep[kpla].get_param().index(paramsplit[2])
        values.append(self.kep[kpla].get_value()[kpar])
      elif paramsplit[0] == 'cov':
        if len(paramsplit) > 2:
          values.append(getattr(self.cov, paramsplit[1])[int(paramsplit[2])])
        else:
          values.append(getattr(self.cov, paramsplit[1]))
      else:
        raise Exception('unknown paramname ({}).'.format(paramname))
    if singlevar:
      return(values[0])
    else:
      return(values)

  def set_params(self, values, paramlist=None):
    if paramlist is None:
      paramlist = self.fitparams
    elif isinstance(paramlist, str):
      paramlist = [paramlist]
      values = [values]
    kwargs = {}
    plapar = [pla.get_value() for pla in self.kep]
    plachange = [False for _ in range(self.npla)]
    for paramname, value in zip(paramlist, values):
      paramsplit = paramname.split('.')
      if paramsplit[0] == 'lin':
        self.linpar[self.linname.index(paramsplit[1])] = value
      elif paramsplit[0] == 'pla':
        kpla = self.planame.index(paramsplit[1])
        kpar = self.kep[kpla].get_param().index(paramsplit[2])
        plapar[kpla][kpar] = value
        plachange[kpla] = True
      elif paramsplit[0] == 'cov':
        if len(paramsplit) > 2:
          if paramsplit[1] not in kwargs:
            kwargs[paramsplit[1]] = getattr(self.cov, paramsplit[1]).copy()
          kwargs[paramsplit[1]][int(paramsplit[2])] = value
        else:
          kwargs[paramsplit[1]] = value
      else:
        raise Exception('unknown paramname ({}).'.format(paramname))
    if len(kwargs) > 0:
      self.cov.update_param(**kwargs)
    for kpla in range(self.npla):
      if plachange[kpla]:
        self.kep[kpla].set_value(plapar[kpla])

  def _sam_logprob(self, x, logprior):
    try:
      lp = logprior(self.fitparams, x)
      if not (lp > -np.inf):
        return(-np.inf)
      ll = self.loglike(x, backup=False)
      if np.isnan(ll):
        return(-np.inf)
      return(lp + ll)
    except:
      return(-np.inf)

  def sample(self, **kwargs):
    x_old = np.array(self.get_params())
    if 'x0' not in kwargs:
      kwargs['x0'] = x_old
    if 'logprob' not in kwargs:
      kwargs['logprob'] = self._sam_logprob
    result = sam(**kwargs)
    self.set_params(x_old)
    return(result)
