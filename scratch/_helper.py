import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# global parameteres
NUM_POSTERIOR_SAMPLES = 5000
NUM_BURNIN_ITERATIONS = 2000
NUM_ADAPTATION = int(0.5 * NUM_BURNIN_ITERATIONS)


def construct_hmc(log_posterior, adaptation_steps=NUM_ADAPTATION):
    hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        # The actual HMC is very simple to define
        tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=log_posterior,    # Log Posterior goes here
            num_leapfrog_steps=3,
            step_size=1                          # constant step size
        ),
        num_adaptation_steps=adaptation_steps
    )
    return hmc


@tf.function
def run_chain(hmc, inits=[], iters=[NUM_POSTERIOR_SAMPLES, NUM_BURNIN_ITERATIONS]):
    samples, is_accepted = tfp.mcmc.sample_chain(
        num_results=iters[0],
        num_burnin_steps=iters[1],
        current_state=inits,
        kernel=hmc
    )
    return samples
