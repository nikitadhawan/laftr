<div align="center">    

# LAFTR

</div>

This repository is an implementation of LAFTR, the Linearized Activation Function TRick, for efficiently estimating function space distances. It reproduces the main experimental results presented in the the paper [Efficient Parametric Approximations of Neural Network Function Space Distance] (https://arxiv.org/abs/2302.03519). 

______________________________________________________________________

## Dependencies

```pip install -r requirements.txt
 ```

______________________________________________________________________

## Continual Learning Experiments

An experiment can be run by defining an `hparams` object in [hp.py](https://github.com/nikitadhawan/laftr/continual/hparams/base_hparams/hp.py) and using the command:

```python -m continual.FSD_CL --hparam_set=hparams
```

Examples of hyperparameter settings that reproduce results in the paper are provided in [hp.py](https://github.com/nikitadhawan/laftr/continual/hparams/base_hparams/hp.py).

______________________________________________________________________

## Influence Function Experiments

Coming soon!

______________________________________________________________________


## Contributors

- [Nikita Dhawan](http://www.cs.toronto.edu/~nikita/)
- [Sheldon Huang](https://www.cs.toronto.edu/~huang/)
- [Juhan Bae](https://www.juhanbae.com/)
