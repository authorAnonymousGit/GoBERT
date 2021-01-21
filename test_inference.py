from inference import predict
import argparse
import torch


if __name__ == '__main__':
    torch.manual_seed(12345)
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter_num", type=int, default=2)
    parser.add_argument("--sub_nn", type=str, default="regular")

    hp = parser.parse_args()

    predict.run_inference(hp.iter_num, hp.sub_nn)

