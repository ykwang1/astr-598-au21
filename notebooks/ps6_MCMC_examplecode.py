import numpy as np
import matplotlib.pyplot as plt

rawfile = np.genfromtxt('data.txt') ##Import data from file location
X = rawfile[:,0] ##X data
Y = rawfile[:,1] ##Y data
Yerr = rawfile[:,2] ##Y error data

params0 = np.array([10.0,2.5]) ## good initial guess for intercept and slope, for example

stepsize = np.array([1.0,0.5]) ## reasonable values for the step size in each parameter, for example

n_steps = 1000000 ## number of steps to run the MCMC

def get_model(params,x): ##evaluate the y-values of the model, given the current guess of parameter values
    
    model = params[0]+params[1]*x ##vector of y-values, given the current guess of parameter values and assumed model form
    
    return model
    
def get_log_likelihood(params, x, y, error): ##obtain the chi2 value of the model y-values given current parameters vs. the measured y-values
    
    chi2 = np.sum(((get_model(params, x)-y)/error)**2) ##calculate chi2

    return chi2
    
def perturb_pick(param): ##select a model parameter to perturb
    
    picker = np.random.randint(0,len(param)) ##this function randomly selects which model parameter to perturb based on how many parameters are in the model
    
    return picker


def propose_param(active_param, stepsize, perturb_value): ##obtain a trial model parameter for the current step
    
    try_param = np.copy(active_param) ##the trial parameter value for this step is based on the value of the parameter in the previous step (copy, so as not to overwrite the previous step value)
    
    step = stepsize*np.random.normal(loc = 0, scale = 1) ##decide how much to perturb the trial parameter based on the stepsize multiplied by a random number drawn from a Gaussian distribution
    
    try_param[perturb_value] = active_param[perturb_value]+step[perturb_value] ##this is the trial value of the parameter for the current step (i.e. the value from the last step plus the perturbation)
    
    return try_param
    
    
def step_eval(params, stepsize, x, y, error, perturb_value): ##evaluate whether to step to the new trial value
    
    chi2_old = get_log_likelihood(params, x, y, error) ##the chi2 value of the parameters from the previous step
    
    try_param = propose_param(params, stepsize, perturb_value) ## read in the trial model parameters for the current step

    chi2_try = get_log_likelihood(try_param, x, y, error) ## the chi2 value of the trial model parameters for the current step
        
    
    if chi2_try < chi2_old: ##if the chi2 value of the trial model parameters < the chi2 value of the parameters from the previous step
        
        new_param = np.copy(try_param) ##accept the trial parameters; the trial parameters become the accepted parameters for this step
        
        acceptvalue = 1 ##record acceptance of the step
        
    else: ##if the chi2 value of the trial model parameters > the chi2 value of the parameters from the previous step
        
        step_prob = np.exp(0.5*(chi2_old-chi2_try)) ##the trial parameters may still be accepted, with some conditional probability based on this formula; the higher the probability, the more likely the trial parameters will be accepted
        
        test_value = np.random.uniform(0,1) ##choose a random value between 0 and 1 to compare against the conditional probability of acceptance of the trial parameters

        if step_prob > test_value: ##if the conditional probability of acceptance of the trial parameters > the random value between 0 and 1
        
            new_param = np.copy(try_param) ##accept the trial parameters; the trial parameters become the accepted parameters for this step
        
            acceptvalue = 1 ##record acceptance of the step
            
        else: ##if the conditional probability of acceptance of the trial parameters < the random value between 0 and 1
            
            new_param = np.copy(params) ##reject the trial parameters; the parameters from the previous step become the accepted parameters for this step
            
            acceptvalue = -1 ##record rejection of the step
        
        
    return new_param, acceptvalue
    

def MCMC(params, stepsize, x, y, error, n_steps): ##run the whole MCMC routine, calling the subroutines
    
    chain = np.zeros((n_steps, len(params))) ##define an empty array to store parameter values for each step in the chain
    
    accept_chain = np.zeros((n_steps, len(params))) ##define an empty array to keep track of acceptance/rejection at each step
    
    chi2_chain = np.zeros(n_steps) ##define an empty array to keep track of the chi2 value at each step in the chain
    
    for i in range(n_steps): ##perform each step
    
        perturb_value = perturb_pick(params) ##select which model value to perturb for this step
    
        params, acceptvalue = step_eval(params, stepsize, x, y, error, perturb_value) ##evaluate whether to step to the new trial value or remain at the parameter values of the previous step
        
        chain[i,:] = params ##store the accepted parameters for this step (i.e. either the new parameter values (if step accepted) or the old parameter values (if step rejected))
        
        accept_chain[i,perturb_value] = acceptvalue ##record acceptance/rejection
        
        chi2_chain[i] = get_log_likelihood(params, x, y, error) ##record chi2 value
        
    return chain, accept_chain, chi2_chain
    
chain, accept_chain, chi2_chain = MCMC(params0, stepsize, X, Y, Yerr, n_steps) ##run the MCMC and output the chains

n_accept_int = len(np.where(accept_chain[:,0] == 1)[0]) ##the number of accepted steps when intercept was perturbed
n_reject_int = len(np.where(accept_chain[:,0] == -1)[0]) ##the number of rejected steps when intercept was perturbed

n_accept_slope = len(np.where(accept_chain[:,1] == 1)[0]) ##the number of accepted steps when slope was perturbed
n_reject_slope = len(np.where(accept_chain[:,1] == -1)[0]) ##the number of rejected steps when slope was perturbed

print( "intercept accept rate: ", 1.0*n_accept_int/(n_accept_int+n_reject_int))
print ("slope accept rate: ", 1.0*n_accept_slope/(n_accept_slope+n_reject_slope))

median_int = np.median(chain[:,0]) ##median value for the model intercept fit
std_int = np.std(chain[:,0]) ##standard deviation for the model intercept fit
median_slope = np.median(chain[:,1]) ##median value for the model slope fit
std_slope = np.std(chain[:,1]) ##standard deviation for the model slope fit
best_params = np.array((median_int,median_slope))

print( "median intercept: ", median_int)
print ("intercept STD: ", std_int)
print ("median slope: ", median_slope)
print ("slope STD: ", std_slope)

