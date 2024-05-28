def get_config(task):
    if task == "timbre_transfer":
        return {
            "output_dir": "timbre_transfer",
            "output_num_files": 1,
            "audio_prompt_file": "piano.wav",
            "audio_prompt_file2": None,
            "ap_ckpt": "pytorch_model.bin",
            "ap_scale": 0.5,
            "time_pooling": 2,
            "freq_pooling": 2,
            "guidance_scale": 7.5,   
            #######################################################
            # You can change the positive_text_prompt whatever you like,
            # But the negative_text_prompt should be the instrument in the 
            # original music.
            "positive_text_prompt": [
                ["a recording of a violin solo"],
                ["a recording of an acoustic guitar solo"],
                ["a recording of a harp solo"]
                ],
            "negative_text_prompt": ["a recording of a piano solo"]
            #######################################################
        }
    elif task == "style_transfer":
        return{
            "output_dir": "style_transfer",
            "output_num_files": 1,
            "audio_prompt_file": "piano.wav",
            "audio_prompt_file2": None,
            "ap_ckpt": "pytorch_model.bin",
            "ap_scale": 0.55,
            "time_pooling": 4,
            "freq_pooling": 4,
            "guidance_scale": 9.5,   
            #######################################################
            ##### You can change the prompt whatever you like #####
            "positive_text_prompt": [
                ["Jazz style music"],
                ["Rock style music"],
                ["Pop style music"]
                ],
            "negative_text_prompt": ["Low quality"]
            #######################################################
        }
    elif task == "accompaniment_generation":
        return {
            "output_dir": "accompaniment_generation",
            "output_num_files": 1,
            "audio_prompt_file": "piano.wav",
            "audio_prompt_file2": None,
            "ap_ckpt": "pytorch_model.bin",
            "ap_scale": 0.5,
            "time_pooling": 2,
            "freq_pooling": 2,
            "guidance_scale": 7.5,   
            #######################################################
            # You can change the positive_text_prompt whatever you like,
            # But the negative_text_prompt should be the instrument in the 
            # original music.
            "positive_text_prompt": [
                ["Duet, Played with violin accompaniment"],
                ["Duet, Played with cello accompaniment"],
                ["Duet, Played with flute accompaniment"]
                ],
            "negative_text_prompt": ["solo"]
            #######################################################
        }
    elif task == "test":
        return {
            "output_dir": "test",
            "output_num_files": 1,
            "audio_prompt_file": "piano.wav",
            "audio_prompt_file2": None,
            "ap_ckpt": "pytorch_model.bin",
            "ap_scale": 0.5,
            "time_pooling": 2,
            "freq_pooling": 2,
            "guidance_scale": 7.5,   
            #######################################################
            # You can change the positive_text_prompt whatever you like,
            # But the negative_text_prompt should be the instrument in the 
            # original music.
            "positive_text_prompt": [""],
            "negative_text_prompt": [""]
            #######################################################
        }
    