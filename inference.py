import torch
from pipeline.pipeline_audioldm2 import AudioLDM2Pipeline
import os
import scipy
# from IPAdapter.ip_adapter.attention_processor import AttnProcessor2_0, CNAttnProcessor2_0,IPAttnProcessor2_0
from APadapter.ap_adapter.attention_processor import AttnProcessor2_0, CNAttnProcessor2_0,IPAttnProcessor2_0
from diffusers.loaders import AttnProcsLayers

sample_rate = 16000
# Hyper parameters
ip_scale = 0.5
time_pooling = 2
freq_pooling = 2
# output folder
dir = "test"
num_files = 5
audio_promt_file2 = None
audio_promt_file = "piano.wav"
ip_ckpt = "/home/fundwotsai/AP-adapter-full/checkpoint-26000/pytorch_model.bin"

os.makedirs(dir, exist_ok=True)
pipeline_trained = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2-large", torch_dtype=torch.float16)
pipeline_trained = pipeline_trained.to("cuda")
layer_num = 0
cross = [None,None,768,768,1024,1024,None,None]
unet = pipeline_trained.unet

# prompt_for_trained = [["country style music"], ["jazz style music"], ["metal style music"], ["reggae style music"],["disco style music"], ["hippop style music"], ["pop style music"], ["rock style music"],]
prompt_for_trained = [["a recording of a Marimba solo"],
                      ["a recording of a violin solo"],
                      ["a recording of a acoustic guitar solo"],
                      ["a recording of a harp solo"]]
# prompt_for_trained = [["Duet, played with Marimba"],
#                       ["Duet, played with piano"],
#                       ["Duet, played with acoustic guitar"],
#                       ["Duet, played with harp"]]
# prompt_for_trained = [["played with an orchestra"], ["a sad music"], ["a happy music"], ["an angry music"]]
negative_prompt_for_trained = [" a recording of a piano solo"]
# negative_prompt_for_trained = ["piano"]

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
                        scale=ip_scale,
                        num_tokens=8,
                        do_copy = False
                    ).to("cuda", dtype=torch.float16)
                #     print("attn_procs",attn_procs)
                else:
                    attn_procs[name] = AttnProcessor2_0()
# # # print("attn_procs",attn_procs)  
state_dict = torch.load(ip_ckpt, map_location="cuda")
# # Iterate through each attention processor
for name, processor in attn_procs.items():
    # Assuming the state_dict's keys match the names of the processors
#     if name in state_dict:
        # Load the weights
        if hasattr(processor, 'to_v_ip') or hasattr(processor, 'to_k_ip'):
                weight_name_v = name + ".to_v_ip.weight"
                weight_name_k = name + ".to_k_ip.weight"
                processor.to_v_ip.weight = torch.nn.Parameter(state_dict[weight_name_v].half())
                processor.to_k_ip.weight = torch.nn.Parameter(state_dict[weight_name_k].half())
                # print(f"weights found for {name}")
        # else:
        #         print(f"Warning: No weights found for {name}")

# # Apply the updated attention processors to the U-Net
unet.set_attn_processor(attn_procs)
# print("attn_procs",attn_procs)

# # this is a hack to synchronize gradients properly. the module that registers the parameters we care about (in
# # this case, AttnProcsLayers) needs to also be used for the forward pass. AttnProcsLayers doesn't have a
# # `forward` method, so we wrap it to add one and capture the rest of the unet parameters using a closure.
class _Wrapper(AttnProcsLayers):
        def forward(self, *args, **kwargs):
                return unet(*args, **kwargs)

unet = _Wrapper(unet.attn_processors)


for i in range(len(prompt_for_trained)):
        waveform = pipeline_trained(audio_file = audio_promt_file, audio_file2 = audio_promt_file2, time_pooling = time_pooling , freq_pooling = freq_pooling, prompt = prompt_for_trained[i]*num_files, negative_prompt = negative_prompt_for_trained*num_files, num_inference_steps=50, guidance_scale = 7.5, num_waveforms_per_prompt=1, audio_length_in_s=10).audios
        for j in range(num_files):
                file_path = os.path.join(dir,f"{prompt_for_trained[i][0]}_{j}_ip{ip_scale}_t{time_pooling}_f{freq_pooling}.wav")
                scipy.io.wavfile.write(file_path, rate=16000, data=waveform[j])

 
