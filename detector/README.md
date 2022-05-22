## Requirments
FNR is built in Python 3.6 using PyTorch 1.8. Please use the following command to install the requirements:

```
pip install -r requirements.txt
```

## How to Run
First, place the data address and configuration into the config file in the data directory, and then follow the train 
and test commands.

### train
To run with Optuna for parameter tuning use this command:

```
python main --data "DATA NAME" --use_optuna "NUMBER OF OPTUNA TRIALS" --batch "BATCH SIZE" --epoch "EPOCHS NUMBER"
```

To run without parameter tuning, adjust your parameters in the config file and then use the below command:
```
python main --data "DATA NAME" --batch "BATCH SIZE" --epoch "EPOCHS NUMBER"
```

### test
In the test step, at first, make sure to have the requested 'checkpoint' file then run the following line:
```
python main --data "DATA NAME" --just_test "REQUESTED TRIAL NUMBER"
```

## Reference
```
@misc{ghorbanpour2021fnr,
      title={FNR: A Similarity and Transformer-Based Approach to Detect Multi-Modal Fake News in Social Media}, 
      author={Faeze Ghorbanpour and Maryam Ramezani and Mohammad A. Fazli and Hamid R. Rabiee},
      year={2021},
      eprint={2112.01131},
      archivePrefix={arXiv},
}
```

