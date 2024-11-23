import torch
import timeit

from methods import tune_hyper_parameter
torch.multiprocessing.set_sharing_strategy('file_system')

# Main function: Tune hyperparameters for the logistic regression model
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start = timeit.default_timer() # start the timer
    target_metric = "acc" or "loss" # set the target metric
    
    # Tune hyperparameters
    best_params, best_metric = tune_hyper_parameter(target_metric, device) 
    stop = timeit.default_timer() # stop the timer
    run_time = stop - start 
    
    print(f"Best {target_metric}: {best_metric:.4f}")
    print(f"Best params:\n{best_params}")

if __name__ == "__main__":
    main()
