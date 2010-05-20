"""
Must have the following in current working directory:
- CSE_Asia_and_Americas...hdf5 (pr-incidence trace)
- pr-falciparum (age-pr relationship trace)
- age-dist-falciparum (age distribution trace)
"""

disttol = 5./6378.
ttol = 1./12

import tables as tb
import numpy as np
from st_cov_fun import *
from generic_mbg import FieldStepper, invlogit, histogram_reduce
from pymc import thread_partition_array
from pymc.gp import GPEvaluationGibbs
import pymc as pm
import mbgw
import os
root = os.path.split(mbgw.__file__)[0]
pm.gp.cov_funs.cov_utils.mod_search_path.append(root)

def check_data(input):
    pass
    
nugget_labels = {'sp_sub': 'V'}
obs_labels = {'sp_sub': 'eps_p_f'}

# Extra stuff for predictive ops.
n_facs = 1000

non_cov_columns = {'lo_age': 'int', 'up_age': 'int', 'pos': 'float', 'neg': 'float'}

# Postprocessing stuff for mapping

def pr(sp_sub):
    pr = sp_sub.copy('F')
    return invlogit(pr)

map_postproc = [pr]
bins = np.array([0,.2])

def binfn(arr, bins=bins):
    out = np.digitize(arr, bins)
    return out

bin_reduce = histogram_reduce(bins,binfn)

def bin_finalize(products, n, bins=bins, bin_reduce=bin_reduce):
    out = {}
    for i in xrange(len(bins)-1):
        out['p-class-%i-%i'%(bins[i]*100,bins[i+1]*100)] = products[bin_reduce][:,i+1].astype('float')/n
    out['most-likely-class'] = np.argmax(products[bin_reduce], axis=1)
    out['p-most-likely-class'] = np.max(products[bin_reduce], axis=1).astype('float') / n
    return out
        
extra_reduce_fns = [bin_reduce]    
extra_finalize = bin_finalize

metadata_keys = ['ti','fi','ui','with_stukel','chunk','disttol','ttol']

# Postprocessing stuff for validation

def pr(data):
    obs = data.pos
    n = data.pos + data.neg
    def f(sp_sub, two_ten_facs=two_ten_factors):
        return pm.flib.invlogit(sp_sub)*two_ten_facs[np.random.randint(len(two_ten_facs))]
    return obs, n, f

validate_postproc=[pr]

def survey_likelihood(x, survey_plan, data, i):
    data_ = np.ones_like(x)*data[i]
    return pm.binomial_like(data_, survey_plan.n[i], pm.invlogit(x))

# Postprocessing stuff for survey evaluation

def simdata_postproc(sp_sub, survey_plan):
    p = pm.invlogit(sp_sub)
    n = survey_plan.n
    return pm.rbinomial(n, p)

# Initialize step methods
def mcmc_init(M):
    M.use_step_method(GPEvaluationGibbs, M.sp_sub, M.V, M.eps_p_f_list, ti=M.ti)

from model import *
