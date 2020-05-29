# Pytorch reimplementation of "A Variational U-Net for Conditional Appearance and Shape Generation"

* original (tensorflow) code is in folder `original`
* pytorch code is in `src`



# Getting the datasets

* refer to the original authors code release to get the datasets.


# Running the model


* The experiment uses the [edflow](https://github.com/pesser/edflow) framework for running experiments.
* The code depends on this [library](https://github.com/theRealSuperMario/supermariopy)
* create a config file as in `configs`
* run with 
```bash
edflow -b configs/xxx.yaml -t -n xxx

# interupt with CTRL + c, then continue with
edflow -b configs/xxx.yaml -t -p xxx
```


# Pretrained Checkpoints

* the authors released original checkpoints [on their page](https://github.com/CompVis/vunet).
* Checkpoints in Pytorch are coming soon.


# TODO:
[ ] implement KL scheduling
[ ] add other datasets, such as Fashion MNIST
