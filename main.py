import argparse
import yaml
from .codes.running import RUNEFD, RUNDENOISE, RUNVIS

parser = argparse.ArgumentParser(description='Main program for opengait.')
parser.add_argument('--cfgs', default="", type=str,
                        help='path of config file')
parser.add_argument('--vis', default="yes", type=str,
                        help='enable visualization')
parser.add_argument('--denoise', default=None, type=str,
                        help='yes / no: overrides denoise_config.enable of the config')
args = parser.parse_args()

def make_efd(cfg):
    obj = RUNEFD(path_cfgs=cfg['paths'], efd_cfgs=cfg['efd_config'])
    obj.run()
    return True

def kill_noise(cfg):
    obj = RUNDENOISE(path_cfgs=cfg['paths'], efd_cfgs=cfg['efd_config'], denoise_cfgs=cfg['denoise_config'])
    obj.run()
    return True

def make_vis(cfg):
    obj = RUNVIS(path_cfgs=cfg['paths'], vis_cfgs=cfg['vis_config'])
    obj.run()
    return True

if __name__ == "__main__":
    with open(args.cfgs, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    check = make_efd(cfg)

    if check:
        denoise = cfg['denoise_config'].get('enable', False) if args.denoise is None else args.denoise == "yes"
        if denoise:
            kill_noise(cfg)

        if args.vis == "yes":
            make_vis(cfg)