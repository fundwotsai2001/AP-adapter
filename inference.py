import torch
from pipeline.pipeline_audioldm2 import AudioLDM2Pipeline
import os
import scipy
from APadapter.ap_adapter.attention_processor import AttnProcessor2_0, CNAttnProcessor2_0, IPAttnProcessor2_0
from diffusers.loaders import AttnProcsLayers
from config import get_config
import argparse

def main(config):
    os.makedirs(config["output_dir"], exist_ok=True)
    
    pipeline_trained = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2-large", torch_dtype=torch.float16)
    pipeline_trained = pipeline_trained.to("cuda")
    layer_num = 0
    cross = [None, None, 768, 768, 1024, 1024, None, None]
    unet = pipeline_trained.unet
    
    positive_text_prompt = config["positive_text_prompt"]
    negative_text_prompt = config["negative_text_prompt"]

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
            cross_attention_dim = cross[layer_num % 8]
            layer_num += 1
            if cross_attention_dim == 768:
                attn_procs[name] = IPAttnProcessor2_0(
                    hidden_size=hidden_size,
                    name=name,
                    cross_attention_dim=cross_attention_dim,
                    scale=config["ap_scale"],
                    num_tokens=8,
                    do_copy=False
                ).to("cuda", dtype=torch.float16)
            else:
                attn_procs[name] = AttnProcessor2_0()

    state_dict = torch.load(config["ap_ckpt"], map_location="cuda")
    for name, processor in attn_procs.items():
        if hasattr(processor, 'to_v_ip') or hasattr(processor, 'to_k_ip'):
            weight_name_v = name + ".to_v_ip.weight"
            weight_name_k = name + ".to_k_ip.weight"
            processor.to_v_ip.weight = torch.nn.Parameter(state_dict[weight_name_v].half())
            processor.to_k_ip.weight = torch.nn.Parameter(state_dict[weight_name_k].half())

    unet.set_attn_processor(attn_procs)

    class _Wrapper(AttnProcsLayers):
        def forward(self, *args, **kwargs):
            return unet(*args, **kwargs)

    unet = _Wrapper(unet.attn_processors)

    for i in range(len(positive_text_prompt)):
        waveform = pipeline_trained(
            audio_file=config["audio_prompt_file"],
            time_pooling=config["time_pooling"],
            freq_pooling=config["freq_pooling"],
            prompt=positive_text_prompt[i] * config["output_num_files"],
            negative_prompt=negative_text_prompt * config["output_num_files"],
            num_inference_steps=50,
            guidance_scale=config["guidance_scale"],
            num_waveforms_per_prompt=1,
            audio_length_in_s=10,
        ).audios
        for j in range(config["output_num_files"]):
            file_path = os.path.join(config["output_dir"], f"{positive_text_prompt[i][0]}_{j}_ip{config['ap_scale']}_t{config['time_pooling']}_f{config['freq_pooling']}.wav")
            scipy.io.wavfile.write(file_path, rate=16000, data=waveform[j])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AP-adapter Inference Script")
    parser.add_argument("--task", type=str, default="style_transfer", help="how do you want to edit the music")
    args = parser.parse_args()  # Parse the arguments
    config = get_config(args.task)  # Pass the parsed arguments to get_config
    print(config)
    main(config)
