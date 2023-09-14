from transformers import AutoConfig, AutoModelForSequenceClassification
import torch.nn as nn


class Deberta(nn.Module):
    """Deberta model for question answering.
    
    Attributes
    ----------
    model: transformers.AutoModelForSequenceClassification
        The model to use.
    config: transformers.AutoConfig
        The config to use.
    """

    def __init__(self, model_base, config):
        """Initialize the deberta model.
        
        Parameters
        ----------
        model_base: str
            The name of the model to use, needs to be a name of type `transformers.AutoModelForSequenceClassification`.
        """

        super(Deberta, self).__init__()
        self.config = AutoConfig.from_pretrained(config)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_base)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                **kwargs):
        """Forward pass of the model.
        
        Parameters
        ----------
        input_ids: torch.Tensor
            The input ids of the input data, shape (batch_size, seq_len),
            where `seq_len` is the maximum length within the batch.
        attention_mask: torch.Tensor
            The attention mask of the input data, shape (batch_size, seq_len)
        token_type_ids: torch.Tensor
            The token type ids of the input data, shape (batch_size, seq_len). Optional.
        
        Returns
        -------
        output: Dict[str: torch.tensor]
        """

        output = self.model(input_ids, attention_mask, token_type_ids)

        return output