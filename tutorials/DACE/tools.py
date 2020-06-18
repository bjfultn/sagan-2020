# -*- coding: utf-8 -*-

# Copyright 2019 Jean-Baptiste Delisle

import numpy as np

def M2E(M, e, ftol=1e-14, Nmax=50):
  """
  Compute eccentric anomaly from mean anomaly (and eccentricity).
  """
  E = M.copy()
  deltaE = np.array([1])
  N = 0
  while max(abs(deltaE))>ftol and N<Nmax:
    diff = M-(E-e*np.sin(E))
    deriv = 1-e*np.cos(E)
    deltaE = diff/deriv
    E += deltaE
    N += 1
  return(E)

def E2v(E, e):
  """
  Compute true anomaly from eccentric anomaly (and eccentricity).
  """
  v = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
  return(v)

def v2E(v, e):
  """
  Compute eccentric anomaly from true anomaly (and eccentricity).
  """
  E = 2*np.arctan(np.sqrt((1-e)/(1+e))*np.tan(v/2))
  return(E)

def E2M(E, e):
  """
  Compute mean anomaly from eccentric anomaly (and eccentricity).
  """
  M = E - e*np.sin(E)
  return(M)

def calc_eM(coefsperio_m_s):
  """
  Compute the eccentricity and mean anomaly (at reference epoch)
  from the coefficients of a linear fit of RV data.
  coefsperio_m_s = [A,B,C,D] with
  rv_m_s = A cos(2pi t/P) + B sin(2pi t/P) + C cos(4pi t/P) + D sin(4pi t/P) + drifts...
  See Delisle et al. (2015)
  """
  # Amplitude of fundamental + first harmonics coefficients
  # WARNING Vk = 2*Vk of Delisle et al. (2015) (eqs 5, 6, 9)
  ampV1_m_s = np.sqrt(coefsperio_m_s[0]**2+coefsperio_m_s[1]**2)
  ampV2_m_s = np.sqrt(coefsperio_m_s[2]**2+coefsperio_m_s[3]**2)
  # Phase of fundamental + first harmonics coefficients
  angV1_rad = np.arctan2(-coefsperio_m_s[1],coefsperio_m_s[0])
  angV2_rad = np.arctan2(-coefsperio_m_s[3],coefsperio_m_s[2])
  # Amplitude of ratio rho (Eq. 14 of Delisle et al. 2015)
  amprhoV = ampV2_m_s/ampV1_m_s
  # Phase of rho (Eq. 14 of Delisle et al. 2015)
  angrhoV_rad = angV2_rad-angV1_rad
  # Very crude estimate of omega (Eq. 28 of Delisle et al. 2015)
  om_rad = 2.0*angV1_rad - angV2_rad
  # Coefficient C(omega) (Eq. 20 of Delisle et al. 2015)
  Com = 1.0/4 * (1-np.exp(-2j*om_rad)/6)
  # Real part of C(omega)
  Rom = 1.0/4 * (1-np.cos(2*om_rad)/6)
  # Estimate of eccentricity (Eq. 24 Delisle et al. 2015)
  if amprhoV + Rom < 1:
    e = 2/np.sqrt(3*Rom) * np.cos((np.pi+np.arccos(3*np.sqrt(3*Rom)*amprhoV/2))/3)
  else:
    e = 0.95
  # Estimate of M0 (Eq. 25 Delisle et al. 2015)
  M0_rad = angrhoV_rad - np.angle(1-e**2*Com)

  return(e, M0_rad)

def designMatrix_Kom(t_d, P_d, e, M0_rad):
  """
  Design matrix for the linear fit of Kcos(om), Ksin(om)
  compute cos(v), sin(v) (v: true anomaly)
  Kcos(om): coef in front of (cos(v)+e)
  Ksin(om): coef in front of -sin(v)
  """
  M_rad = M0_rad + 2*np.pi*t_d/P_d
  E_rad = M2E(M_rad,e)
  v_rad = E2v(E_rad,e)
  return(np.array([np.cos(v_rad)+e,-np.sin(v_rad)]))

def computeSecAcc(plx_mas, pma_maspyr, pmd_maspyr):
  """
  Compute the secular acceleration from parallax and proper motion.
  """
  d_m = 1000.0/plx_mas*3.08567758e16
  mu_radps = np.sqrt(pma_maspyr*pma_maspyr+pmd_maspyr*pmd_maspyr)*2*np.pi/(
    360.0*1000.0*3600.0*86400.0*365.25)
  return(d_m*mu_radps*mu_radps*86400.0)

def read_rdb(filename):
  """
  Read a rdb file
  and return the dictionnary of its columns (as numpy arrays).
  """

  with open(filename, 'r') as rdbfile:
    header = rdbfile.readline()
  data = np.genfromtxt(filename, skip_header=2,
    comments='#', dtype=None, encoding=None)
  datadic = {}
  if len(data.shape) == 2:
    for colid, colname in enumerate(header.split()):
      datadic[colname] = data[:,colid]
  else:
    for colid, colname in enumerate(header.split()):
      datadic[colname] = np.array([line[colid] for line in data],
        dtype=data.dtype[colid])
  return(datadic)

def smooth_series(renorm_time, series, kernel):
  series_smoothed = np.empty_like(series)
  kdef = ~np.isnan(series)
  for k in range(renorm_time.size):
    w = kernel(renorm_time-renorm_time[k])
    w /= np.sum(w[kdef])
    series_smoothed[k] = np.sum(w[kdef]*series[kdef])
  return(series_smoothed)

def gaussian_kernel(x):
  """
  Gaussian kernel for the smooth_series function
  x = delta t / tau
  """
  return(np.exp(-0.5*x*x))

def box_kernel(x):
  """
  Box kernel for the smooth_series function
  x = delta t / width
  """
  return(np.abs(x)<=1.0)

def epanechnikov_kernel(x):
  """
  Epanechnikov kernel for the smooth_series function
  x = delta t / width
  """
  return((np.abs(x)<=1.0)*(1.0-x*x))
