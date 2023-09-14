# %%
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AutoConfig
import torch.nn as nn
import torch


class DistilBert(nn.Module):
    """DistilBert model for question answering.
    Attributes
    ----------
    model : transformers.AutoModelForQuestionAnswering
        The model to use.
    config : transformers.AutoConfig
        The config to use.
    """

    def __init__(self, base_name: str, **kwargs):
        """Initializes the distil ber model.

        Parameters
        ----------
        base_name : str
            The name of the model to use, needs to be a name of type `transformers.AutoModelForQuestionAnswering`.
        """
        super(DistilBert, self).__init__()
        self.config = AutoConfig.from_pretrained(base_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            base_name, config=self.config)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                start_positions=None,
                end_positions=None,
                **kwargs):
        """Forward pass of the model.
        Parameters
        ----------
        input_ids : torch.Tensor
            The input ids of the input data, shape (batch_size, seq_len),
            where `seq_len` is the maximum length within the batch.
        attention_mask : torch.Tensor
            The attention mask of the input data, shape (batch_size, seq_len)
        token_type_ids : torch.Tensor
            The token type ids of the input data, shape (batch_size, seq_len). Optional.
        start_positions : torch.Tensor
            The start positions of the answer spans in the batch data, shape (batch_size).
             Only passed in when training or validating (and loss is calculated).
        end_positions : torch.Tensor
            The end positions of the answer spans in the batch data, shape (batch_size).
            Only passed in when training or validating (and loss is calculated).
        Returns
        -------
        y : Dict[str:torch.tensor]
            with keys `loss`, `start_logits`, `end_logits`:
            loss : shape ()
            start_logits : shape (batch_size, seq_len)
            end_logits : shape (batch_size, seq_len)
        """

        y = self.model(input_ids,
                       attention_mask,
                       token_type_ids,
                       start_positions=start_positions,
                       end_positions=end_positions)
        return y
