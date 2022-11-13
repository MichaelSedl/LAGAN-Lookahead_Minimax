import argparse
from evaluator import Evaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("wdir", type=str)
    parser.add_argument(
        "--eval_model",
        type=str,
        default="all",
        choices=["gen", "gen_avg", "gen_ema", "gen_ema_slow", "all"],
    )
    parser.add_argument("--eval_pth", type=str, default="all")
    parser.add_argument("--cont", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    evaluator = Evaluator(args.wdir, args.eval_model, args.eval_pth)
    evaluator.eval_metrics(args.cont)
    # wdir = "./results/cifar10/gan_adam_resnet/G0.000200_D0.000200_beta_0.0_bs_128_dIters_5"
    # eval_model = "all"
    # eval_pth = "iter00009999.pth"
    # eval_pth = "all"
