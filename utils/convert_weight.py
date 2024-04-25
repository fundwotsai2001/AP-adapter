import torch
from safetensors import safe_open

tensors = {}
with safe_open("/home/fundwotsai/DreamSound/audioldm2-large-ipadapter-all-cfg/pipeline_step_50/unet/diffusion_pytorch_model.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
ckpt = "/home/fundwotsai/DreamSound/audioldm2-large-ipadapter-all-cfg/pipeline_step_50/unet/diffusion_pytorch_model.safetensors"
sd = torch.load(ckpt, map_location="cpu")
image_proj_sd = {}
ip_sd = {}
print(sd)
for k in sd:
    # print(k)
    # print(sd[k])
    if k.startswith("unet"):
        
        ip_sd[k.replace("unet.", "")] = sd[k]
    elif k.startswith("image_proj_model"):
        image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
    # elif k.startswith("adapter_modules"): 
    #     # ip_sd[k.replace("adapter_modules.", "")] = sd[k]
        

torch.save({"unet": ip_sd}, "ip_adapter.bin")