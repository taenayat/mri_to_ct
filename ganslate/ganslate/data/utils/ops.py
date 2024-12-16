# import numpy as np


# def pad(volume, target_shape):
#     assert len(target_shape) == len(volume.shape)
#     # By default no padding
#     pad_width = [(0, 0) for _ in range(len(target_shape))]

#     for dim in range(len(target_shape)):
#         if target_shape[dim] > volume.shape[dim]:
#             pad_total = target_shape[dim] - volume.shape[dim]
#             pad_per_side = pad_total // 2
#             pad_width[dim] = (pad_per_side, pad_total % 2 + pad_per_side)

#     return np.pad(volume, pad_width, 'constant', constant_values=volume.min())

import torch

def pad(volume, target_shape):
    assert len(target_shape) == len(volume.shape)
    
    # Default no padding
    pad_width = []
    
    for dim in range(len(target_shape) - 1, -1, -1):  # PyTorch's pad expects reversed order
        if target_shape[dim] > volume.shape[dim]:
            pad_total = target_shape[dim] - volume.shape[dim]
            pad_per_side = pad_total // 2
            pad_width.extend([pad_per_side, pad_total % 2 + pad_per_side])
        else:
            pad_width.extend([0, 0])
    # print(pad_width)
    padded_volume = torch.nn.functional.pad(
        volume, pad_width, mode='constant', value=volume.min().item()
    )
    return padded_volume


def dynamic_pad_collate(batch):
    # Initialize lists for each key
    A_batch = [item['A'] for item in batch]

    # Determine max dimensions for 'A'
    max_depth = max(item.shape[1] for item in A_batch)
    max_height = max(item.shape[2] for item in A_batch)
    max_width = max(item.shape[3] for item in A_batch)
    desired_shape = (max_depth, max_height, max_width)

    padded_A = []
    padded_B = []
    masks = []
    metadata = []

    for item in batch:
        padded_A_item = pad(item['A'].squeeze(0), desired_shape)
        padded_A.append(padded_A_item.unsqueeze(0))

        if 'B' in item:
            padded_B_item = pad(item['B'].squeeze(0), desired_shape)
            padded_B.append(padded_B_item.unsqueeze(0))

        if 'masks' in item:
            mask_batch = {}
            for key, mask in item['masks'].items():
                padded_mask = pad(mask.squeeze(0), desired_shape)
                mask_batch[key] = padded_mask.unsqueeze(0)
            masks.append(mask_batch)

        if 'metadata' in item:
            metadata.append(item['metadata'])

    batch_dict = {
        'A': torch.stack(padded_A)
        # 'B': torch.stack(padded_B)
    }
    if padded_B:
        batch_dict['B'] = torch.stack(padded_B)

    if masks:
        collated_masks = {}
        for mask_batch in masks:
            for key, mask in mask_batch.items():
                if key not in collated_masks:
                    collated_masks[key] = []
                collated_masks[key].append(mask)
        for key in collated_masks:
            collated_masks[key] = torch.stack(collated_masks[key])
        batch_dict['masks'] = collated_masks

    if metadata:
        collated_metadata = {}
        for meta in metadata:
            for key, value in meta.items():
                if key not in collated_metadata:
                    collated_metadata[key] = []
                collated_metadata[key].append(value)
        batch_dict['metadata'] = collated_metadata

    return batch_dict