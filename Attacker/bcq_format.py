import torch


def b_mtx_to_flat(b_mtx, num_bits=3):
    b = b_mtx.detach().cpu().float()

    if b.ndim == 2 and b.shape[1] == num_bits:
        return b

    if b.ndim >= 1 and b.shape[0] == num_bits:
        return b.movedim(0, -1).reshape(-1, num_bits)

    if b.ndim >= 1 and b.shape[-1] == num_bits:
        return b.reshape(-1, num_bits)

    raise ValueError(f"Unsupported B_mtx shape: {tuple(b.shape)}")


def weighted_sum_to_shape(b_mtx, bit_weights, target_shape, num_bits=3):
    b_flat = b_mtx_to_flat(b_mtx, num_bits=num_bits)
    bit_weights = bit_weights.to(dtype=b_flat.dtype)
    return (b_flat @ bit_weights).reshape(target_shape)
