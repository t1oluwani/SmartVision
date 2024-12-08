import requests
from collections import OrderedDict

lr_response = requests.post("http://localhost:5000/train/LR")
lr_data = lr_response.json()
lr_results = OrderedDict(
    train_accuracy=lr_data["train_accuracy"],
    test_accuracy=lr_data["test_accuracy"],
    average_loss=lr_data["average_loss"],
    run_time=lr_data["run_time"],
)

fnn_response = requests.post("http://localhost:5000/train/FNN")
fnn_data = fnn_response.json()
fnn_results = OrderedDict(
    train_accuracy=fnn_data["train_accuracy"],
    test_accuracy=fnn_data["test_accuracy"],
    average_loss=fnn_data["average_loss"],
    run_time=fnn_data["run_time"],
)

cnn_response = requests.post("http://localhost:5000/train/CNN")
cnn_data = cnn_response.json()
print(cnn_data)
cnn_results = OrderedDict(
    train_accuracy=cnn_data["train_accuracy"],
    test_accuracy=cnn_data["test_accuracy"],
    average_loss=cnn_data["average_loss"],
    run_time=cnn_data["run_time"],
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