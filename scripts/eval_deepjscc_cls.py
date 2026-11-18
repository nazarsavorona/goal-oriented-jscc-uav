"""Single deepjscc codec DINOv2 cls-sim @ given SNR (proven loader from eval_r1_paired).
Usage: eval_deepjscc_cls.py <ckpt> <valdir> <n> <snr> [codec_size]"""
import glob
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from transformers import AutoModel

from semvideo.channel.awgn import AWGNChannel
from semvideo.models.deep_jscc import DeepJSCC

ckpt, valdir, n, SNR = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
sz = int(sys.argv[5]) if len(sys.argv) > 5 else 256
net = DeepJSCC(channel_ratio=1/6, base_channels=64).cuda().eval()
sd = torch.load(ckpt, map_location="cpu", weights_only=False)
state = sd.get("model_state_dict", sd.get("model_state", sd)) if isinstance(sd, dict) else sd
miss, unexp = net.load_state_dict(state, strict=False)
assert not miss and not unexp, f"arch mismatch: {len(miss)} missing, {len(unexp)} unexpected"
ch = AWGNChannel(snr_db=SNR).cuda().eval()
dino = AutoModel.from_pretrained("facebook/dinov2-small").cuda().eval()
mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).cuda()
std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).cuda()
@torch.no_grad()
def cls(x):
    x = F.interpolate(x.clamp(0,1),(518,518),mode="bilinear",align_corners=False)
    return dino(pixel_values=(x-mean)/std).last_hidden_state[:,0]
paths = sorted(glob.glob(valdir+"/*.jpg")+glob.glob(valdir+"/*.png"))
idx = np.linspace(0,len(paths)-1,min(n,len(paths)),dtype=int)
paths = [paths[int(i)] for i in idx]
sims=[]
with torch.no_grad():
    for p in paths:
        bgr = cv2.imread(p)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2,0,1).float().unsqueeze(0).cuda()/255.
        t = F.interpolate(t,(sz,sz),mode="bilinear",align_corners=False)
        rec = net(t, ch)
        c0, c1 = cls(t), cls(rec.clamp(0,1))
        sims.append(F.cosine_similarity(c0,c1).item())
a=np.array(sims)
name=ckpt.split("/")[-2]
print("CLSSIM ckpt=%s snr=%s n=%d cls_mean=%.5f cls_std=%.5f"%(name,SNR,len(a),a.mean(),a.std()))
