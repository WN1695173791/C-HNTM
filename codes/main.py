import os
import traceback
import argparse
from run import *

parser = argparse.ArgumentParser("NTM")
parser.add_argument("--data", type=str, default="20news")
parser.add_argument("--n_topic", type=int, default=100)
parser.add_argument("--n_cluster", type=int, default=20)
parser.add_argument("--n_epoch", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--momentum", type=float, default=0.9)

args = parser.parse_args()



def main():
    mode = "train"
    # mode = "load"
    if mode == "train":
        runner = C_HNTM_Runner(args, mode=mode)
        try:
            runner.train()
        except KeyboardInterrupt as e:
            print("KeyboardInterrupt... Aborted.")
        except Exception as e:
            print(traceback.print_exc())
    elif mode == "load":
        model_path = "../models/c_hntm/c_hntm_2022-04-13-15.pkl"
        runner = C_HNTM_Runner(args, mode=mode)
        runner.load(model_path)
        runner.show_topic_results()




if __name__ == "__main__":
    main()