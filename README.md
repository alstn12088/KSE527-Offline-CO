## Project

This is official PyTorch code for KSE527 final project held in 2022 sprining somestar in KAIST. 


## Source Code Reference

This code is originally implemented based on  [Attention Model](https://github.com/wouterkool/attention-learn-to-route) , which is source code of the paper   [Attention, Learn to Solve Routing Problems!](https://openreview.net/forum?id=ByxBFsRqYm) which has been accepted at [ICLR 2019](https://iclr.cc/Conferences/2019), cite as follows:

```
@inproceedings{
    kool2018attention,
    title={Attention, Learn to Solve Routing Problems!},
    author={Wouter Kool and Herke van Hoof and Max Welling},
    booktitle={International Conference on Learning Representations},
    year={2019},
    url={https://openreview.net/forum?id=ByxBFsRqYm},
}
```

Our work revised learning scheme of Attention Model (AM), improving sample efficiency in offline CO. The most of code configuration is same, except:

* Modified "net/attention_model.py" for state forcing with guiding state, and log-likelihood caluation of guiding action. 
* Modified "train.py" to import offline data, to perform label augmentation and to perform self-supervised learning (psuedo label based behavior cloning). 

We remarked "KSE527" to revised part in the source code . 


## How to Use

### Unzip data 

We provide pre-collected labeled data (from Concorde) which is pre-processed for training in "data/offline_labels.pkl".
We provide pre-defined problem set which is corresponding to the labeled data. 
We provide benchmark test data in "data/tsp20_test_seed1234.pkl"



### Evaluation with pretrained model

```bash
python eval.py --dataset_path data/tsp20_test_seed1234.pkl --model pretrained ours/tsp20/epoch-99.pt
```

```bash
python eval.py --dataset_path data/tsp100_test_seed1234.pkl --model pretrained ours/tsp100/epoch-99.pt
```

### Training the Model

#### Training AM with our method (AM + Data augmentation + Self-supervised Learning) 

```bash
python run.py --graph_size 20 --training_model ssl 
```

#### Training AM with data augmentation(AM + Data augmentation) 

```bash
python run.py --graph_size 20 --training_model da
```

#### Training AM with naive offline behavior cloning (baseline) 

```bash
python run.py --graph_size 20 --training_model bc
```

```

#### GPU

Only a single GPU is available in this code. 


### Other usage

```bash
python run.py -h
python eval.py -h
```



## Dependencies

* Python>=3.8
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.1
* tqdm
* [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger)
* Matplotlib 

