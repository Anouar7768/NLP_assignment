import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel



class TransformerBinaryClassifier(torch.nn.Module):

    def __init__(self, plm_name: str):
        super(TransformerBinaryClassifier, self).__init__()
        self.lmconfig = AutoConfig.from_pretrained(plm_name)
        self.lmtokenizer = AutoTokenizer.from_pretrained(plm_name, add_special_tokens=True)
        self.lm = AutoModel.from_pretrained(plm_name, output_attentions=False)
        self.emb_dim = self.lmconfig.hidden_size
        self.output_size = 3
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim, self.output_size),
            torch.nn.Softmax()
        )
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x):
        """
        Function to do forward of the neural network
        """
        x: torch.Tensor = self.lm(x['input_ids'], x['attention_mask']).last_hidden_state
        cls_vects = x[:, 0, :]  # extract the [CLS] token of each sequence
        x = self.classifier(cls_vects)
        return x.squeeze(-1)

    def compute_loss(self, predictions, target):
        """
        Compute the loss
        """
        return self.loss_fn(predictions, target)


