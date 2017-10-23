import tensorflow as tf
import math
import os

def logsum_mog(x, pi, mu1, mu2, sigma1, sigma2):
	return log_sum_exp(tf.log(pi) + log_normal(x, mu1, sigma1),
	                   tf.log(1. - pi) + log_normal(x, mu2, sigma2))


def log_sum_exp(u, v):
	m = tf.maximum(u, v)
	return m + tf.log(tf.exp(u - m) + tf.exp(v - m))


def log_normal(x, mu, sigma):
	return -0.5 * tf.log(2.0 * math.pi) - tf.log(tf.abs(sigma)) - tf.square((x - mu)) / (
		2 * tf.square(sigma))


def compute_KL(shape, mu, sigma, prior, sample):
	"""
	Compute KL divergence between posterior and prior.
	"""
	posterior = tf.contrib.distributions.Normal(mu, sigma)
	KL = tf.reduce_sum(posterior.log_prob(tf.reshape(sample, [-1])))
	N1 = tf.contrib.distributions.Normal(0.0, prior.sigma1)
	N2 = tf.contrib.distributions.Normal(0.0, prior.sigma2)
	mix1 = tf.reduce_sum(N1.log_prob(sample), 1) + tf.log(prior.pi_mixture)
	mix2 = tf.reduce_sum(N2.log_prob(sample), 1) + tf.log(1.0 - prior.pi_mixture)
	prior_mix = tf.stack([mix1, mix2])
	KL += -tf.reduce_sum(tf.reduce_logsumexp(prior_mix, [0]))
	return KL
