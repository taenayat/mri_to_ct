import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    print("GPU MEMORY:")
    print('memory allocated', torch.cuda.memory_allocated())
    print('memory reserved', torch.cuda.memory_reserved())
