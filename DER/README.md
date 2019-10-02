## Requirements

- Python 2.7
- TensorFlow 1.8
- numpy 1.14


## Project Structure

    .
    ├── conf                    # Config files
    ├── data
        ├── der_data            # Experiment data   
    ├── results                 # Results saving
    ├── model          
        ├── DER.py              # The graph of DER
    ├── data_loader.py          
    ├── evaluate.py             # Evaluation code
    ├── main.py                 # The entrance of the project
    └── solver.py               # The solver of the project     




## Config

example: 

[path]

root_path = your-local-path/DER

input_data_type = data/der_data/category

output_path = results

log_conf_path = conf/logging.conf


[parameters]

global_dimension = 8

word_dimension = 64

batch_size = 50

epoch = 50

learning_rate = 0.001

reg = 1

mode = validation

merge = FM

concat = 1

item_review_combine = add

item_review_combine_c = 0.5

lmd = 1

drop_out_rate = 0.7


## Usage

1. Install all the required packages

2. process the raw data into the formats according to data/der_data/data_format

3. Run python main.py


## Author# DER
