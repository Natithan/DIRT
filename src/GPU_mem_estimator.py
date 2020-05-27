from my_utils.model_utils import sizeof_fmt, sz


def pretty_mem(num_elements,element_size=4):
    return sizeof_fmt(num_elements*element_size)

def drop(els):
    # From testing in code: applying dropout to tensor adds 1/4th of the tensor size to mem again :P
    # When debugging slow, adds the same + that 4th again for some reason ...
    # return els +els/4
    return els / 4
d_hidden = 768
d_ff = 4*d_hidden
d_seq=128
d_batch = 4
d_float = 4 #size in bytes
nb_heads=12
nb_layers = 12
h_activations = d_hidden*d_seq*d_batch
ff_h_activations = d_ff*d_seq*d_batch
att_els = d_seq*d_seq*d_batch*nb_heads
D = 2 #rel top_down level

def bertlayer_elmts():
    elmts = 0
    # Attention
    elmts += 3 * h_activations  # project q, k and v
    elmts += 3 * h_activations  #reshape q k and v
    elmts += 2*att_els + drop(att_els) #attention weights: make, softmax, dropout
    elmts += 4 * h_activations + drop(h_activations) #attention output: make, reshape, project, dropout, layernorm

    #FF
    elmts += ff_h_activations*2 + h_activations + drop(h_activations) #make ff_h, activate,make h_, drop h
    elmts += drop(h_activations) + h_activations # drop h, layernorm
    return elmts


def dirtlayer_elmts():
    elmts = 0
    elmts += h_activations  # make masked activations
    elmts += D*bertlayer_elmts()
    elmts += 2 * h_activations  # make left and right rolled
    elmts += 3*(ff_h_activations*2+h_activations) #ff_h , activate and h_ for topdown, left, right
    elmts += 3 * h_activations + h_activations # concat and project to h_
    elmts += (d_batch + 1) * h_activations # roll for each batch, and again for positive
    return elmts

def per_layer_elmts():
    elmts = 0
    elmts += dirtlayer_elmts() # self-predicting
    elmts += bertlayer_elmts() #clean forward pass
    return elmts


elmts = 0
elmts += per_layer_elmts()*nb_layers
print(pretty_mem(elmts,d_float))