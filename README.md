# Bayesian Recurrent Neural Networks

This is a replication of the paper 'Bayesian Recurrent Neural Networks' by Meire Fortunato, Charles Blundell, Oriol Vinyals.

Paper: https://arxiv.org/abs/1704.02798

Status: Basic model replicated.

# Requirements
- Python 3.5
- [TensorFlow 1.3.0](https://www.tensorflow.org/)

# Usage
    $ sh download_ptb.sh
    $ python bayesian_rnn.py -model medium -log_sigma1 -1.0 -log_sigma2 -7.0 -prior_pi 0.25

### To-do:
- Implement posterior sharpening.
- Implement image captioning experiment.

### Acknowledgements

Thanks to Meire Fortunato for providing the Bayes by Backprop/cell code and @alexkrk for an initial implementation.
