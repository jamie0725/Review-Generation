# Explainable recommendation with review generation

### About
This repository contains code for our report Prototype-Enhanced Explainable Recommendation with Synthetic Reviews.
![](arch.jpg)

### Description
We improve upon the [MRG](https://github.com/PreferredAI/mrg) model in the following aspects:
1. adding extra input features by embedding the history reviews of given (item, user) pair using GRU (we denote as the prototype).
2. adding attention layers to pick out the most 'useful' information while embedding the history reviews for generating syntactic reviews.

### Dataset
We use the same [data](http://static.preferred.ai/mrg/data.zip) as the original MRG paper to make the results directly comparable.

### Dependencies
- Python 3
- TensorFlow >=1.12,<2.0
- Hickle
- Tqdm
- [GloVe](https://nlp.stanford.edu/projects/glove/) word embeddings

### Running Instructions
```bash
python train.py --data_dir ./data --batch_size 64 --learning_rate 0.001 --num_epochs 20
```
Training arguments:
```bash
python train.py --help
```
```
optional arguments:
  -h, --help                show this help message and exit
  --data_dir                DATA_DIR
                            Path to the data directory
  --learning_rate           LEARNING_RATE
                            Learning rate (default: 3e-4)
  --dropout_rate            DROPOUT_RATE
                            Probability of dropping neurons (default: 0.2)
  --lambda_reg              LAMBDA_REG
                            Lambda hyper-parameter for regularization (default: 1e-4)
  --num_epochs              NUM_EPOCHS
                            Number of training epochs (default: 20)
  --batch_size              BATCH_SIZE
                            Batch size of reviews (default: 64)
  --num_factors             NUM_FACTORS
                            Number of latent factors for users/items/prototypes (default: 256)              
  --word_dim                WORD_DIM
                            Word embedding dimensions (default: 200)
  --lstm_dim                LSTM_DIM
                            Hidden dimensions of the LSTM Cell (default: 256)
  --max_length              MAX_LENGTH
                            Maximum length of reviews to be generated (default: 20)
  --display_step            DISPLAY_STEP
                            Display info after number of steps (default: 10)
  --allow_soft_placement    ALLOW_SOFT_PLACEMENT
                            Allow device soft device placement
```