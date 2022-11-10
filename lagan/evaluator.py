import os
import ast
import json
import torch
import tensorflow.compat.v1 as tf
import numpy as np
import fid
import utils

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
        self.eval_pth = {}
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

    def eval_metrics(self):
        utils.make_folder(self.metrics_path)
        for m in self.eval_pth:
            jf = os.path.join(self.metrics_path, f"{m}.json")
            for pth in self.eval_pth[m]:
                self.load_gen(m, pth)
                print(f"Evaluating: {m}, {self.iter}")
                FID = self.eval_fid()
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
    E.eval_metrics()
