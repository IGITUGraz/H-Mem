# H-Mem: Harnessing synaptic plasticity with Hebbian Memory Networks
This is the code used in the paper "[H-Mem: Harnessing synaptic plasticity with Hebbian Memory
Networks](https://www.biorxiv.org/content/10.1101/2020.07.01.180372v1)" for training H-Mem on a single-shot
image association task and on the bAbI question-answering tasks.

![H-Mem schema](https://i.imgur.com/fK3UWaP.png)

## Setup
You need [TensorFlow](https://www.tensorflow.org/) to run this code. We tested it on TensorFlow version 2.1.
Additional dependencies are listed in [environment.yml](environment.yml). If you use
[Conda](https://docs.conda.io/en/latest/), run

```bash
conda env create --file=environment.yml
```

to install the required packages and their dependencies.

## Usage
Run the following command

```bash
python babi_task_single.py
```

to start training on bAbI task 1 in the 10k training examples setting. Set the command line argument `--task_id` to train on other tasks. You can try different model configurations by changing various command line arguments. For example,

```bash
python babi_task_single.py --task_id=4 --memory_size=20 --epochs=50 --logging=1
```

will train the model with an associative memory of size 20 on task 4 for 50 epochs. The results will be stored in `/results`.

## References
* Limbacher, T., Legenstein, R. (2020). H-Mem: Harnessing synaptic plasticity with Hebbian Memory Networks bioRxiv https://dx.doi.org/10.1101/2020.07.01.180372
