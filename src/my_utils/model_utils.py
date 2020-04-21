import torch
from torch import nn

from config import FLAGS


def apply_sequence_mask(given):
    """Replaces each slice of the given tensor ALONG THE SEQUENCE DIMENSION (ASSUMED 1) with zeroes with a probability FLAGS.masking_fraction
    Also returns the mask
    """

    mask = (torch.rand(given.shape[1]) > FLAGS.masking_fraction).cuda(
        given.device)  # Same mask for items in batch
    broadcast_ready_mask = mask[None, :, None]
    masked_in_state = given * broadcast_ready_mask
    return masked_in_state, mask


def masked_MSE_loss(target, predicted, mask):
    '''
    Returns a mean-square-error loss that only considers sequence elements (along the 2nd dimension) for which the mask is zero
    '''
    return torch.mean((((target - predicted) * ~mask[None, :, None]) ** 2))


def contrastive_L2_loss(in_state, predicted_in_state, mask):
    if FLAGS.d_batch <= 1:
        raise ValueError('Using DIR requires batch size bigger than 1 to contrast with')
    d_batch = in_state.shape[0]
    negative_loss = sum([masked_MSE_loss(in_state.roll(shifts=i, dims=0), predicted_in_state, mask) for i in
                         range(
                             d_batch)]) / d_batch if d_batch > 1 else torch.tensor(
        1.)
    # Positive loss: distance to corresponding batch element
    positive_loss = masked_MSE_loss(in_state, predicted_in_state, mask)
    layer_loss = positive_loss / negative_loss
    return layer_loss


def process_targets_for_loss(target_tokens):
    max_target_seq_length = target_tokens.shape[-1]  # Longest length if no adjacent masks
    current_target_seq_length = target_tokens.shape[1]
    padding_index = 0
    padder = nn.ConstantPad1d((0, max_target_seq_length - current_target_seq_length), padding_index)
    target_tokens_contiguous = padder(target_tokens).contiguous().view(-1)

    return target_tokens_contiguous

def get_activation():
    if FLAGS.activation == 'relu':
        return nn.ReLU()
    elif FLAGS.activation == 'gelu':
        return nn.GELU()
    else:
        raise ValueError('Unsupported activation provided')


def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def sizeof_mb(num, suffix='B'):
    for unit in ['','Ki']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Mi', suffix)



def sz(tensor_or_module):
    if isinstance(tensor_or_module, torch.Tensor):
        tensor = tensor_or_module
        return sizeof_fmt(tensor.nelement() * tensor.element_size())
    elif isinstance(tensor_or_module, nn.Module):
        module = tensor_or_module
        return sizeof_fmt(sum(p.numel() * p.element_size() for p in module.parameters() if p.requires_grad))
    else:
        raise ValueError(f"Cannot determine size for argument of type {type(tensor_or_module)}")