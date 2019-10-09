## How to run
Create a virtualenv with `python2`
First install the required dependencies: `pip install -r requirements.txt`  
Download the datasets and word vectors from https://worksheets.codalab.org/worksheets/0x1ad3f387005c492ea913cf0f20c9bb89/ and store them in the same directory as this.  
Run the code from this directory using the command `export COPY_EDIT_DATA=$(pwd); export PYTHONIOENCODING=utf8; cd cond-editor-codalab; python train_ctx_vae.py`