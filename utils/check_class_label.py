import json
import os
labels_count = {}
filename = 'labels_count.json'
from collections import OrderedDict

for root, dirs, files in os.walk("/home/fundwotsai/DreamSound/Fast-Audioset-Download/wavs/eval"):
            for file in files:
                if file.endswith('.json'):
                    json_path = os.path.join(root, file)
                    with open(json_path, 'r') as f:
                        metadata = json.load(f)
                        # print([key for key in metadata.keys()])
                        # audio_path = os.path.join('/home/fundwotsai/DreamSound/Fast-Audioset-Download', metadata['path'])
                        # check_wav_file(audio_path)
                        if 'labels' in metadata:
                            labels = metadata['labels']
                            for label in labels:
                                labels_count[label] = labels_count.get(label, 0) + 1
                        else:
                            print(f"'labels' key not found in file: {json_path}")

                        # for label in labels:
                        #     labels_count[label] = labels_count.get(label, 0) + 1


# Writing JSON data
sorted_labels_count = sorted(labels_count.items(), key=lambda x: x[1], reverse=True)

# Convert to OrderedDict to preserve order
ordered_labels_count = OrderedDict(sorted_labels_count)

# Writing JSON data with custom formatting
with open(filename, 'w') as f:
    for key, value in ordered_labels_count.items():
        json.dump({key: value}, f)
        f.write('\n')