import torch
from allennlp.training import GradientDescentTrainer
from allennlp.nn import util as nn_util


class MyTrainer(GradientDescentTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)


    def batch_loss(self, batch, for_training):
        """
        Does a forward pass on the given batches and returns the `loss` value in the result.
        If `for_training` is `True` also applies regularization penalty.
        """
        batch = nn_util.move_to_device(batch, self.cuda_device)
        if isinstance(batch,torch.Tensor):
            output_dict = self._pytorch_model(batch)
        else:
            output_dict = self._pytorch_model(**batch)


        try:
            loss = output_dict["loss"]
            if for_training:
                loss += self.model.get_regularization_penalty()
        except KeyError:
            if for_training:
                raise RuntimeError(
                    "The model you are trying to optimize does not contain a"
                    " 'loss' key in the output of model.forward(inputs)."
                )
            loss = None

        return loss
