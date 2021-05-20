from inference import predict
import argparse
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters_num", type=int, default=1)
    hp = parser.parse_args()
    for item_num in range(1, hp.iters_num + 1):
        predict.run_inference(item_num, "regular")
