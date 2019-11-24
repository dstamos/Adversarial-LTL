# Online-Within-Online Meta-Learning - NeurIPS 2019
We study the problem of learning a series of tasks in a fully online Meta-Learning
setting. The goal is to exploit similarities among the tasks to incrementally adapt
an inner online algorithm in order to incur a low averaged cumulative error over
the tasks. We focus on a family of inner algorithms based on a parametrized
variant of online Mirror Descent. The inner algorithm is incrementally adapted
by an online Mirror Descent meta-algorithm using the corresponding within-task
minimum regularized empirical risk as the meta-loss. In order to keep the process
fully online, we approximate the meta-subgradients by the online inner algorithm.
An upper bound on the approximation error allows us to derive a cumulative
error bound for the proposed method. Our analysis can also be converted to the
statistical setting by online-to-batch arguments. We instantiate two examples of the
framework in which the meta-parameter is either a common bias vector or feature
map. Finally, preliminary numerical experiments confirm our theoretical findings.

## Requirements


### Installing cvxpy
https://www.cvxpy.org/install/

```
pip install cvxpy
```

### Citation

```
@inproceedings{deveni2019owo,
  title={Online-Within-Online Meta-Learning.},
  author={Deveni, Giulia and Stamos, Dimitris and Ciliberto, Carlo and Pontil, Massimiliano},
  booktitle={Conference on Neural Information Processing Systems},
  year={2019}
}
```
