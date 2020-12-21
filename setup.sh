
# create venv
python3 -m venv ecg_venv
# activate venv
source ecg_venv/bin/activate
# install required libs
pip install -r requirements.txt

# unzip sample raw data
# your pwd must be ecg_classification_challenge
tar -xvf data/raw/train.tar
tar -xvf data/raw/validation.tar
tar -xvf data/raw/test.tar
