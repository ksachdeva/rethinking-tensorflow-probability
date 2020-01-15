from scratch._helper import run_chain
from scratch._helper import construct_hmc

import arviz
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def get_dataset():
    d = pd.read_csv(
        'data/Howell1.csv', sep=';', header=0)
    d2 = d[d.age >= 18]
    return d2


df = get_dataset()


def main_ber_coroutine():

    def joint_co_berno():
        prior_rate = yield Root(tfd.Uniform(0., 1.))
        rv_data = yield tfd.Bernoulli(probs=prior_rate)

    jdc_berno = tfd.JointDistributionCoroutine(
        joint_co_berno, validate_args=True)

    generator = tfd.Bernoulli(probs=0.7)
    data = generator.sample(1000)

    def unnormalized_berco(rate):
        return jdc_berno.log_prob(rate, data)

    results = run_chain(construct_hmc(
        unnormalized_berco), inits=[0.1])

    print(results)


def main_regular():

    # actual rate/prob is 0.7
    generator = tfd.Bernoulli(probs=0.7)
    data = generator.sample(1000)

    def joint_log_prob(data, rate):
        prior_rate = tfd.Uniform(0., 1.)
        rv_data = tfd.Bernoulli(probs=rate)

        return (
            prior_rate.log_prob(rate) +           # the prior
            tf.reduce_sum(rv_data.log_prob(data))  # the likelihood:
        )

    def unnormalized_posterior(rate):
        return joint_log_prob(data, rate)

    results = run_chain(construct_hmc(
        unnormalized_posterior), inits=[0.1])

    print(results)


def main():
    main_regular()


if __name__ == '__main__':
    main()
