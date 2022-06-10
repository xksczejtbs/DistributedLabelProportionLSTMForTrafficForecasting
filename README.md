# Distributed Label Proportion LSTM for Traffic Forecasting

main: `geometrictemporaltorch.py`

datadir path for cpu: `params.py`

datadir path for gpu: top of `geometrictemporaltorch.py`

config of parameters: `geometrictemporaltorch.py` in dict config at the end of the file

run: python geometrictemporaltorch.py

start tensorboard with: tensorboard --logdir ~/geotorchtemporal/ray_results/yyyy-mm-dd_h-m-s.ms_gitsha/ --port 6006 
