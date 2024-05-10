"""
HHR ops from https://arxiv.org/pdf/2109.02157.pdf
"""
import copy

import torch
from torch.distributions import Normal
#from torch.fft import fft, ifft


def fft(x):
    return torch.fft.rfft(x, norm='ortho')


def ifft(x):
    return torch.fft.irfft(x, norm='ortho')


def bind(a, b):
    return torch.real(ifft(torch.multiply(fft(a), fft(b))))


def unbind(s, a):
    return bind(s, inverse(a))


def inverse(a):
    a = torch.flip(a, dims=[-1])
    return torch.roll(a, 1, dims=-1)


# def unit_projection(a, eps=1e-8):
#     a_hat = fft(a)
#     a_hat = a_hat / (a_hat.abs() + eps)
#     return torch.real(ifft(a_hat))


def unit_projection(x):
    c = fft(x)
    c_ish = c / torch.norm(c, dim=-1, keepdim=True)
    output = ifft(c_ish)
    return torch.real(output)


def init(shape):
    a = torch.randn(*shape) / shape[-1]
    return unit_projection(a)


def init_ortho(shape):
    """
    Generate n vectors of size dims that are orthogonal to each other.
    """
    num_vectors, dims = shape
    # Intializing class vectors.
    vecs = torch.randn(dims, num_vectors, dtype=torch.float)

    # Using QR decomposition to get orthogonal vectors.
    vecs, _ = torch.qr(vecs)
    vecs = vecs.t()
    vecs = vecs / torch.norm(vecs, dim=-1, keepdim=True)
    return vecs


def unit_regularization(v):
    v_hat = fft(v)
    v_hat = v_hat * torch.norm(v_hat, dim=-1, keepdim=True)
    x = torch.real(ifft(v_hat))
    dist = Normal(0., 1. / v.shape[-1])
    nlp = -dist.log_prob(x)
    return nlp


def key_value_query(
    k: torch.Tensor, v: torch.Tensor, q: torch.Tensor,
    causal: bool = True, norm: bool = False
):
    #k, v, inv_q = fft(k), fft(v), inverse(fft(q))
    k, v, inv_q = fft(k), fft(v), fft(q)
    if norm:
        eps = 1e-8
        k = k / (torch.norm(k, dim=-1, keepdim=True) + eps)
        v = v / (torch.norm(v, dim=-1, keepdim=True) + eps)
        inv_q = inv_q / (torch.norm(inv_q, dim=-1, keepdim=True) + eps)
    kv = torch.multiply(k, v)
    if causal:
        r = kv.cumsum(dim=-2) #* kv.shape[-1] / kv.shape[-2]
    else:
        r = kv.sum(dim=-2, keepdim=True)
    # unbind values for each query
    qv = torch.real(ifft(torch.multiply(r, inv_q)))
    # if norm:
    #     eps = 1e-8
    #     qv = qv / (torch.norm(qv, dim=-1, keepdim=True) + eps)

    return qv


def perm_key_value_query(
    k: torch.Tensor, v: torch.Tensor, q: torch.Tensor, perms: torch.Tensor,
    causal: bool = True,
):
    """
    Create a key-value vector and then retrieve queried values using HRR.

    This function is meant to reduce the number of fft/ifft calls compared to naively
    binding k/v, summing over the sequence, and then unbinding q.
    """
    # NOTE: perhaps we can avoid explicitly inverting and assume inv_q is learned by the model?
    # k, v, inv_q = fft(k), fft(v), inverse(fft(q))
    k, v, inv_q = fft(k), fft(v), fft(q)
    inv_q = inv_q[..., perms].permute(2, 0, 1, 3)
    k = k[..., perms].permute(2, 0, 1, 3)
    v = v[None, ...]
    kv = k * v
    if causal:
        r = kv.cumsum(dim=-2) #* kv.shape[-1] / kv.shape[-2]
    else:
        r = kv.sum(dim=-2, keepdim=True)
    # unbind values for each query/permutation and take the mean
    qv = (r * inv_q).mean(0)
    qv = torch.real(ifft(qv))
    return qv


def rebind(kv, a, b, do_fft=True):
    """Unbinds key a and then rebinds the value with key b"""
    a_inv = inverse(a)
    if do_fft:
        kv, a, a_inv, b = map(fft, (kv, a, a_inv, b))

    # result = kv + new_key_kv - old_key_kv
    # result = kv + b * av - a * av
    # result = kv + av * (b - a)
    result = kv + kv * a_inv * (b - a)

    if do_fft:
        result = torch.real(ifft(result))
    return result


def transform(kv, a, f, do_fft=True):
    """Unbinds key a and then rebinds the same key with a transformed value"""
    a_inv = inverse(a)
    if do_fft:
        kv, a, a_inv = map(fft, (kv, a, a_inv))

    v = kv * a_inv
    vs = v.shape
    flat_real_shape = vs[:-1] + (vs[-1] * 2,)
    v_real = torch.view_as_real(v)
    real_shape = v_real.shape
    v_real = v_real.view(*flat_real_shape)
    fv = f(v_real)
    fv = torch.view_as_complex(fv.view(real_shape))
    # result = kv + new_key_kv - old_key_kv
    # result = kv + a * tv - a * v
    # result = kv + a * (tv - v)
    result = kv + a * (fv - v)

    if do_fft:
        result = torch.real(ifft(result))
    return result


def transform_rebind(kv, a, b, f, do_fft=True):
    """Unbinds key a and then binds the transformed value to key b"""
    a_inv = inverse(a)
    if do_fft:
        kv, a, a_inv, b = map(fft, (kv, a, a_inv, b))

    a_value = kv * a_inv
    # result = kv + new_kv - old_kv
    result = kv + b * f(a_value.view_as) - a * a_value
    if do_fft:
        result = torch.real(ifft(result))
    return result


def wrap_real_transform(f, arg):
    """
    View a complex tensor `arg` as real-valued with final dimension doubled in size then apply
    a function `f` and reshape result to complex values by halving size of last dimension.
    """
    flat_real_shape = arg.shape[:-1] + (arg.shape[-1] * 2,)
    arg_real = torch.view_as_real(arg)
    real_shape = arg_real.shape
    arg_real = arg_real.view(*flat_real_shape)
    result = f(arg_real)
    result = torch.view_as_complex(result.view(real_shape))
    return result