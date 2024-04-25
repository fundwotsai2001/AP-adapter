import torch
from pipeline.pipeline_audioldm2 import AudioLDM2Pipeline
import os
import scipy
from IPAdapter.ip_adapter.attention_processor import AttnProcessor2_0, CNAttnProcessor2_0,IPAttnProcessor2_0
from diffusers.loaders import AttnProcsLayers

save_weight_dir = "copied_cross_attention"
ip_ckpt = "audioldm2-large/unet/diffusion_pytorch_model.bin"
os.makedirs(save_weight_dir, exist_ok=True)
pipeline_trained = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2-large", torch_dtype=torch.float16)
pipeline_trained = pipeline_trained.to("cuda")
layer_num = 0
cross = [None,None,768,768,1024,1024,None,None]
unet = pipeline_trained.unet
attn_procs = {}
for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor2_0()
        else:
                cross_attention_dim = cross[layer_num%8]
                layer_num = layer_num + 1
                if cross_attention_dim == 768:
                    attn_procs[name] = IPAttnProcessor2_0(
                        hidden_size=hidden_size,
                        name = name,
                        cross_attention_dim=cross_attention_dim,
                        scale=0.5,
                        num_tokens=8,
                    ).to("cuda", dtype=torch.float16)
                #     print("attn_procs",attn_procs)
                else:
                    attn_procs[name] = AttnProcessor2_0()
# # # print("attn_procs",attn_procs)  
state_dict = torch.load(ip_ckpt, map_location="cuda")
for key in state_dict.keys():
    print(key)
# # Iterate through each attention processor
for name, processor in attn_procs.items():
    # Assuming the state_dict's keys match the names of the processors
#     if name in state_dict:
        # Load the weights
        if hasattr(processor, 'to_v_ip') or hasattr(processor, 'to_k_ip'):
            name = name.split(".")[:-1]
            name = ".".join(name)
            print(name)
            weight_name_v = name + ".to_v.weight"
            weight_name_k = name + ".to_k.weight"
            processor.to_v_ip.weight = torch.nn.Parameter(state_dict[weight_name_v].half())
            processor.to_k_ip.weight = torch.nn.Parameter(state_dict[weight_name_k].half())
            print(f"weights found for {name}")
            torch.save(processor.to_v_ip.weight, f'{save_weight_dir}/{name}.processor_v.bin')
            torch.save(processor.to_k_ip.weight, f'{save_weight_dir}/{name}.processor_k.bin')
            print(f"Weights saved for {name} to_v_ip and to_k_ip")



 
