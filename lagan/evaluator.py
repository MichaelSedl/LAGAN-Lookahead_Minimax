import os
import time
import datetime
import ast
import json
import torch
import tensorflow.compat.v1 as tf
import numpy as np
import fid
import inception_score
import utils
from collections import defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class Evaluator(object):
    def __init__(self, wdir, eval_model="all", eval_pth="all"):
        self.wdir = wdir
        self.setup_config()
        # aggregate models and pth files for evaluation
        if eval_model == "all":
            self.eval_model = next(os.walk(os.path.join(self.wdir, "models")))[1]
        else:
            self.eval_model = [eval_model]
        if not self.lookahead:
            try:
                self.eval_model.remove("gen_ema_slow")
            except ValueError:
                pass
        self.eval_pth = defaultdict(list)
        for m in self.eval_model:
            if eval_pth == "all":
                self.eval_pth[m] = sorted(
                    os.listdir(os.path.join(self.wdir, "models", m))
                )
            else:
                self.eval_pth[m] = [eval_pth]

        self.setup_gen()
        self.setup_inception_network()

    def setup_config(self):
        log_file = os.path.join(self.wdir, "logs", "log.log")
        configs = []
        prefix = "Namespace("
        with open(log_file) as f:
            for ln in f:
                if ln.startswith(prefix):
                    configs.append(ln.lstrip(prefix).rstrip().rstrip(")"))
        config_string = configs[-1].split(", ")
        for kv in config_string:
            k, v = kv.split("=")
            setattr(self, k, ast.literal_eval(v))

    def setup_gen(self):
        if self.arch == "resnet":
            from models import resnet_models

            self.G = resnet_models.Generator(self.z_dim).cuda()
        else:
            raise NotImplementedError

    def setup_inception_network(self):
        sample_noise = torch.FloatTensor(self.sample_size_fid, self.z_dim).normal_()
        self.fid_noise_loader = torch.utils.data.DataLoader(
            sample_noise, batch_size=200, shuffle=False
        )

        _INCEPTION_PTH = fid.check_or_download_inception(
            "./precalculated_statistics/inception-2015-12-05.pb"
        )
        print("Loading the Inception Network from: {}".format(_INCEPTION_PTH))
        fid.create_inception_graph(
            _INCEPTION_PTH
        )  # load the graph into the current TF graph
        _gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        # _gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4)
        self.fid_session = tf.Session(config=tf.ConfigProto(gpu_options=_gpu_options))
        print("Loading real data FID stats from: {}".format(self.fid_stats_path))
        _real_fid = np.load(self.fid_stats_path)
        self.mu_real, self.sigma_real = _real_fid["mu"][:], _real_fid["sigma"][:]
        _real_fid.close()

    def load_gen(self, model, pth_file):
        self.G.load_state_dict(
            torch.load(os.path.join(self.wdir, "models", model, pth_file))
        )
        self.iter = int(pth_file.lstrip("iter").rstrip(".pth"))

    def eval_fid(self):
        self.G.eval()
        fake_samples = np.empty((self.sample_size_fid, self.imsize, self.imsize, 3))
        for j, noise in enumerate(self.fid_noise_loader):
            noise = noise.cuda()
            i1 = j * 200  # batch_size = 200
            i2 = i1 + noise.size(0)
            samples = self.G(noise).cpu().data.add(1).mul(255 / 2.0)
            fake_samples[i1:i2] = samples.permute(0, 2, 3, 1).numpy()
        self.G.train()
        mu_g, sigma_g = fid.calculate_activation_statistics(
            fake_samples, self.fid_session, batch_size=100
        )
        fid_score = fid.calculate_frechet_distance(
            mu_g, sigma_g, self.mu_real, self.sigma_real
        )
        return fid_score

    def eval_is(self):
        all_samples = []
        samples = torch.randn(self.sample_size_fid, self.z_dim)
        for i in range(0, self.sample_size_fid, 200):
            samples_100 = samples[i:i+200]
            samples_100 = samples_100.cuda()
            all_samples.append(self.G(samples_100).cpu().data.numpy())

        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = np.multiply(np.add(np.multiply(all_samples, 0.5), 0.5), 255).astype('int32')
        all_samples = all_samples.reshape((-1, 3, self.imsize, self.imsize)).transpose(0, 2, 3, 1)
        return inception_score.get_inception_score(list(all_samples))

    def eval_metrics(self, continue_evaluation=False):
        utils.make_folder(self.metrics_path)
        if continue_evaluation:
            for mjsonl in os.listdir(self.metrics_path):
                evaluated_iter = []
                with open(os.path.join(self.metrics_path, mjsonl)) as f:
                    for ln in f:
                        evaluated_iter.append(json.loads(ln)["iter"])
                evaluated_iter = sorted(np.unique(evaluated_iter))
                evaluated_pth = [f"iter{ev_it:08d}.pth" for ev_it in evaluated_iter]
                m = mjsonl.replace(".jsonl", "")
                remaining_pth = [pth for pth in self.eval_pth[m] if pth not in evaluated_pth]
                self.eval_pth[m] = remaining_pth

        for m in self.eval_pth:
            jf = os.path.join(self.metrics_path, f"{m}.jsonl")
            start_time = time.time()
            for pth in self.eval_pth[m]:
                self.load_gen(m, pth)
                print(f"Evaluating: {m}, {self.iter}")

                FID = self.eval_fid()
                FID = float(FID)
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print(f"FID evaluated: {m}, {self.iter}; Elapsed [{elapsed}]")

                IS_mean, IS_std = self.eval_is()
                IS_mean = float(IS_mean)
                IS_std = float(IS_std)
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print(f"\nIS evaluated: {m}, {self.iter}; Elapsed [{elapsed}]")
                print(f"FID: {FID:.4f}, IS: {IS_mean:.4f} +- {IS_std:.4f}\n")
                with open(jf, "a") as fs:
                    s = json.dumps(
                        dict(iter=self.iter, FID=FID, IS_mean=IS_mean, IS_std=IS_std,)
                    )
                    fs.write(f"{s}\n")


if __name__ == "__main__":
    wdir = (
        "./results/cifar10/gan_adam_resnet/G0.000200_D0.000200_beta_0.0_bs_128_dIters_5"
    )
    eval_model = "gen"
    # eval_model = "all"
    eval_pth = "iter00009999.pth"
    # eval_pth = "all"

    E = Evaluator(wdir, eval_model, eval_pth)
#    E.load_gen(eval_model, eval_pth)
#    print(E.eval_fid())
#    print(E.eval_is())
    E.eval_metrics()
