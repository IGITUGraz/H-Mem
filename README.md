# H-Mem: Harnessing synaptic plasticity with Hebbian Memory Networks
This is the code used in the paper "[H-Mem: Harnessing synaptic plasticity with Hebbian Memory
Networks](https://www.biorxiv.org/content/10.1101/2020.07.01.180372v2)" for training H-Mem on a single-shot
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

### Single-shot associations with H-Mem
To start training on the single-shot image association task, run

```bash
python image_association_task.py
```

Set the command line argument `--delay` to set the between-image delay (in the paper we used delays ranging from 0 to 40). Run the following command

```bash
python image_association_task_lstm.py
```

to start training the LSTM model on this task (the default value for the between-image delay is 0; you can change it with the command line argument `--delay`).

### Question answering with H-Mem
Run the following command

```bash
python babi_task_single.py
```

to start training on bAbI task 1 in the 10k training examples setting. Set the command line argument `--task_id` to train on other tasks. You can try different model configurations by changing various command line arguments. For example,

```bash
python babi_task_single.py --task_id=4 --memory_size=20 --epochs=50 --logging=1
```

will train the model with an associative memory of size 20 on task 4 for 50 epochs. The results will be stored in `results/`.

### Memory-dependent memorization
In our extended model we have added an 'read-before-write' step. This model will be used if the
command line argument `--read_before_write` is set to `1`. Run the following command

```bash
python babi_task_single.py --task_id=16 --epochs=250 --read_before_write=1
```

to start training on bAbI task 16 in the 10k training examples setting (note that we trained the extended
model for 250 epochs---instead of 100 epochs). You should get an accuracy of about 100% on this task. Compare
to the original model, which does not solve task 16, by running the following command

```bash
python babi_task_single.py --task_id=16 --epochs=250
```

## References
* Limbacher, T., & Legenstein, R. (2020). H-Mem: Harnessing synaptic plasticity with Hebbian Memory Networks. Advances in Neural Information Processing Systems, 33.
https://www.biorxiv.org/content/10.1101/2020.07.01.180372v2
