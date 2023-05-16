import torch
from pdb import set_trace


def packbits_padded(tensor, dim=-1, mask=0b1, out=None, dtype=torch.uint8):
    dim = dim if dim >= 0 else dim + tensor.dim()
    nbits_element, nbits = 8, (
        1 if mask == 0b00000001 else 2 if mask == 0b00000011 else 4 if mask == 0b00001111 else 8 if mask == 0b11111111 else None)
    nibbles = nbits_element // nbits
    assert tensor.shape[dim] % nibbles == 0, "shape: {}, dim: {}, nibbles: {}".format(tensor.shape, dim, nibbles)

    out = out if out is not None else torch.empty(*tensor.shape[:dim], tensor.shape[dim] // nibbles,
                                                  *tensor.shape[1 + dim:], dtype=dtype, device=tensor.device)
    shift = torch.arange(nbits_element - nbits, -1, -nbits, dtype=torch.uint8, device=tensor.device)
    shift = shift.view(nibbles, *((1,) * (tensor.dim() - dim - 1)))
    torch.sum(tensor.view(*tensor.shape[:dim], -1, nibbles, *tensor.shape[1 + dim:]) << shift, dim=1 + dim, out=out)
    return out


def unpackbits_padded(tensor, dim=-1, mask=0b1, out=None):
    dim = dim if dim >= 0 else dim + tensor.dim()
    nbits_element, nbits = 8, (
        1 if mask == 0b00000001 else 2 if mask == 0b00000011 else 4 if mask == 0b00001111 else 8 if mask == 0b11111111 else None)
    nibbles = nbits_element // nbits

    out = out if out is not None else \
        torch.empty(*tensor.shape[:dim], tensor.shape[dim] * nibbles, *tensor.shape[1 + dim:], dtype=torch.uint8,
                    device=tensor.device)
    shift = torch.arange(nbits_element - nbits, -1, -nbits, dtype=torch.uint8, device=tensor.device)
    shift = shift.view(nibbles, *((1,) * (tensor.dim() - dim - 1)))
    torch.bitwise_and((tensor.unsqueeze(1 + dim) >> shift).view_as(out), mask, out=out)
    return out

def sparsify(tensor, mask, with_batch_size=False):
    shape = tensor.shape
    shape = torch.tensor(shape)

    mask = mask.reshape(-1)
    sparse = tensor.reshape(-1)[mask]
    if with_batch_size:
        sparse = sparse.reshape(shape[0], -1)
    else:
        sparse = sparse.unsqueeze(0)

    # add bits to make it divisible by 8
    if mask.shape[0] % 8 != 0:
        add_bits = 8 - (mask.shape[0] % 8)
        mask = torch.cat([mask, torch.zeros(add_bits, dtype=mask.dtype, device=mask.device)], dim=0)

    mask = packbits_padded(mask)

    # idle value
    # mask = torch.ones(1, device=tensor.device)
    # sparse = tensor

    return shape, mask, sparse


def unsparsify(shape, mask, sparse, with_batch_size=False):
    mask = unpackbits_padded(mask).to(dtype=torch.bool)
    if with_batch_size:
        sparse = sparse.view(-1)
    else:
        sparse = sparse.squeeze(0)

    shape = torch.Size(shape)
    dense = torch.zeros(shape.numel(), device=sparse.device, dtype=sparse.dtype)
    dense[mask[:shape.numel()]] = sparse

    return dense.reshape(shape)

    # idle
    # return sparse


if __name__ == "__main__":
    with torch.no_grad():
        attn = torch.rand(1, 12, 195, 195)
        # attn = torch.rand(1, 1, 6, 6)
        mask = attn > 0.5
        attn_sparse = mask * attn

        shape_save, mask_save, sparse_save = sparsify(attn_sparse, mask)
        attn_sparse_2 = unsparsify(shape_save, mask_save, sparse_save)

        print("torch.norm(attn_sparse-attn_sparse_2) is {}".format(torch.norm(attn_sparse-attn_sparse_2)))
