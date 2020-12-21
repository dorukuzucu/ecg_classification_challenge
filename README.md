# ecg_classification_challenge

## Install

Install the requirements (this may take a few minutes).
```
./setup.sh
```

## Preprocesing
To preprocess the raw data use following command.

```
python src/data/preprocess_data.py
```

## Training

To train a model use the following command, updating the config paremeters in `demo.py`:

```
python demo.py
```

## Testing
After training the model for a few epochs, you can make predictions with.
To do so you need to update the model path which created under `results/ecg_net_results`

```
python src/model/evaluate.py
```


Credits to this repo [@dorukuzucu](https://github.com/dorukuzucu) and [@osmandemirel](https://github.com/osmandemirel).
