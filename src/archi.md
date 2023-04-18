## Model

Based on the "Adapt or Get Left Behind:Domain Adaptation through BERT Language Model Finetuning for Aspect-Target Sentiment Classification" article by Alexander Rietzler et al. https://arxiv.org/pdf/1908.11860.pdf. 

Model : pre-trained **BERT base uncased**. 

## Pre-processing :

Raw data : Sentence - Aspect term - Aspect category.
  
Input to the tokenizer : "sentence. [SEP] aspect term [SEP] Aspect category"

Output of the tokenizer : "[CLS] token_sentence [SEP] token_aspect_term [SEP] token_aspect_category [SEP]".

No need to troncate the input data. 
  
## Classifier

  The classification task will be performed on top of the [CLS] token hidden representation, used for sequence-pair classification.
  
