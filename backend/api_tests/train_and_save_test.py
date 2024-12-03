import requests
from collections import OrderedDict

cnn_response = requests.post("http://localhost:5000/train/CNN")
cnn_data = cnn_response.json()
cnn_results = OrderedDict(
    accuracy=cnn_data["accuracy"],
    run_time=cnn_data["run_time"],
)

fnn_response = requests.post("http://localhost:5000/train/FNN")
fnn_data = fnn_response.json()
fnn_results = OrderedDict(
    accuracy=fnn_data["accuracy"],
    run_time=fnn_data["run_time"],
)

lr_response = requests.post("http://localhost:5000/train/LR")
lr_data = lr_response.json()
lr_results = OrderedDict(
    accuracy=lr_data["accuracy"],
    run_time=lr_data["run_time"],
)

print("CNN Results:")
for key in cnn_results:
    print(f"\t{key}: {cnn_results[key]}")
    
print("FNN Results:")   
for key in fnn_results:
    print(f"\t{key}: {fnn_results[key]}")
    
print("LR Results:")
for key in lr_results:
    print(f"\t{key}: {lr_results[key]}")


