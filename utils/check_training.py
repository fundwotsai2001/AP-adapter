import torch
import os

# Assuming unet_1 and unet_2 are already loaded as dictionaries
unet_1_path = "/home/fundwotsai/DreamSound/audioldm2-large-ipadapter-audioset-unet-random-pooling_v2/checkpoint-1/pytorch_model.bin"
unet_2_path = "/home/fundwotsai/DreamSound/audioldm2-large-ipadapter-audioset-unet-random-pooling_v2/checkpoint-2/pytorch_model.bin"

unet_1 = torch.load(unet_1_path, map_location="cpu")
unet_2 = torch.load(unet_2_path, map_location="cpu")
print(unet_1)
import torch
import os

def sum_elements_in_tensors(directory):
    total_sum = 0
    for filename in os.listdir(directory):
        if filename.endswith('.bin'):
            file_path = os.path.join(directory, filename)
            tensor = torch.load(file_path)

            # Summing all elements in the tensor
            tensor_sum = tensor.sum().item()
            # Adding to the total sum
            total_sum += tensor_sum

    return total_sum

directory = '/home/fundwotsai/DreamSound/save_attention_weight'
with torch.no_grad():  # Disable gradient computation
    total_sum = sum_elements_in_tensors(directory)
print(f"Total Sum Across All Tensors: {total_sum}")



# Replace 'path_to_directory' with your directory's path

# Do something with total_parameters, like printing or saving
def calculate_dict_sum(model_dict):
    total_sum = 0
    num = 0
    for param_tensor in model_dict.values():
        total_sum += torch.sum(param_tensor)
        num += 1
    return total_sum, num

# Calculate the sum of all values in each dictionary
unet_1_sum, unet_1_num = calculate_dict_sum(unet_1)
unet_2_sum, unet_2_num = calculate_dict_sum(unet_2)

# Print the sums
print(f"Sum of values in unet_1: {unet_1_sum},{unet_1_num}")
print(f"Sum of values in unet_2: {unet_2_sum},{unet_2_num}")

# Check if the sums are different
are_sums_different = unet_1_sum != unet_2_sum
print(f"Are the sums of the values different? {are_sums_different}")
