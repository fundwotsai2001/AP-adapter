import torch
from pipeline.pipeline_audioldm2 import AudioLDM2Pipeline
import os
import scipy
import argparse
from APadapter.ap_adapter.attention_processor import AttnProcessor2_0, CNAttnProcessor2_0, IPAttnProcessor2_0
from diffusers.loaders import AttnProcsLayers

def main(args):
    os.makedirs(args.dir, exist_ok=True)
    
    pipeline_trained = AudioLDM2Pipeline.from_pretrained("cvssp/audioldm2-large", torch_dtype=torch.float16)
    pipeline_trained = pipeline_trained.to("cuda")
    layer_num = 0
    cross = [None, None, 768, 768, 1024, 1024, None, None]
    unet = pipeline_trained.unet

    prompt_for_trained = [
        ["a recording of a Marimba solo"],
        ["a recording of a violin solo"],
        ["a recording of an acoustic guitar solo"],
        ["a recording of a harp solo"]
    ]
    negative_prompt_for_trained = ["a recording of a piano solo"]

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
                    scale=args.ap_scale,
                    num_tokens=8,
                    do_copy=False
                ).to("cuda", dtype=torch.float16)
            else:
                attn_procs[name] = AttnProcessor2_0()

    state_dict = torch.load(args.ap_ckpt, map_location="cuda")
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

    for i in range(len(prompt_for_trained)):
        waveform = pipeline_trained(
            audio_file=args.audio_prompt_file,
            audio_file2=args.audio_prompt_file2,
            time_pooling=args.time_pooling,
            freq_pooling=args.freq_pooling,
            prompt=prompt_for_trained[i] * args.num_files,
            negative_prompt=negative_prompt_for_trained * args.num_files,
            num_inference_steps=50,
            guidance_scale=7.5,
            num_waveforms_per_prompt=1,
            audio_length_in_s=10
        ).audios
        for j in range(args.num_files):
            file_path = os.path.join(args.dir, f"{prompt_for_trained[i][0]}_{j}_ip{args.ap_scale}_t{args.time_pooling}_f{args.freq_pooling}.wav")
            scipy.io.wavfile.write(file_path, rate=16000, data=waveform[j])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AudioLDM2 Inference Script")
    parser.add_argument("--dir", type=str, default="test", help="Output directory")
    parser.add_argument("--num_files", type=int, default=5, help="Number of files to generate per prompt")
    parser.add_argument("--audio_prompt_file", type=str, default="piano.wav", help="Path to the primary audio prompt file")
    parser.add_argument("--audio_prompt_file2", type=str, default=None, help="Path to the secondary audio prompt file")
    parser.add_argument("--ap_ckpt", type=str, default="pytorch_model.bin", help="Path to the AP checkpoint file")
    parser.add_argument("--ap_scale", type=float, default=0.5, help="AP scale")
    parser.add_argument("--time_pooling", type=int, default=2, help="Time pooling factor")
    parser.add_argument("--freq_pooling", type=int, default=2, help="Frequency pooling factor")

    args = parser.parse_args()
    main(args)
