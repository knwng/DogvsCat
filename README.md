# Dog&Cat image classifier for course: Introduction to Artificial Intelligence

## Requirement

---

- Python 2.7
- Tensorflow 1.4.1

Requirements can be installed by

```bash
pip install -r requirements.txt
```

## Dataset

---

​	We use dataset from [here](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data), you may also want to use some external data like [The Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/)

​	If you want to use your own dataset, make sure you have a data list containing the image path and label as follows:

```
dataset/train/cat.1.jpg 0
dataset/train/dog.2.jpg 1
```

​	The script *./dataset/create_dataset.py* can help you create such list and split data into train and val by randomly sampling or K-fold

```bash
./create_dataset.py \
--data_split_type k-fold \
--fold_num 10 \
--labelmap ./label_map.txt \
--data_dir ./dataset/train
```

## Pretrained Model

---

​	We use ImageNet pretrained [Inception-ResNet-v2](https://arxiv.org/abs/1602.07261) from [Tensorflow official repository](https://github.com/tensorflow/models/tree/master/research/slim), you need to download the checkpoint file as long as the code. You need to put the checkpoint file under ./pretrained\_model like: 

```
./pretrained_model/inception_resnet_v2.ckpt
```

​	If you want to use other pretrained backbone models, you need to put the network defination code under ./nets and prepare the checkpoint

## Train

---

​	After datasets and pretrained model are prepared, you can train the model just run

```bash
./train.py \
--train_dataset dataset/train.txt \
--val_dataset dataset/val.txt \
--train_dir experiments/expr1 \
--learning_rate 1e-4 \
--epoch 100 \
--batch_size 32 \
--image_size 224 \
--pretrained_model ./pretrained_model/inception_resnet_v2.ckpt
```

​	If you want to resume training, just add ` --resume` to the command above



## Evaluation

---

​	There are several tools provided to evaluate the model.

###### Evaluation using accuracy, precision and recall

You can use `eval.py` to evaluate your model using val data and calculate the accuracy, precision and recall, the command is:

```bash
./eval.py \
--val_dataset dataset/val.txt \
--train_dir experiments/expr1 \
--checkpoint model-10000 \
--batch_size 128
```

## Prediction

---

If you want to generate submission for the *Dogs vs Cats Competition*, you can use `test.py` :

```bash
./test.py \
--test_dataset dataset/test.txt \
--train_dir experiments/expr1 \
--checkpoint model-10000 \
--batch_size 128
```

If you have several models, you can use simple ensembling technique to improve your performance:

```bash
# Assume your submissions are placed as: 
# experiments/expr1-1/submission.csv
# experiments/expr1-2/submission.csv
# experiments/expr1-3/submission.csv
# ...

./ensemble.py \
--fold_num 10 \
--ensembled_root experiments/expr1 \
--submission_name submission.csv
```

