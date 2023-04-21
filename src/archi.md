## 1. Name of the students

Pierre Boscherel, Alexis-Raja Brachet, Laurane Gorgues, Anouar Oussalah

## 2. Description of the classifier

The overall architecture is based on the "Adapt or Get Left Behind:Domain Adaptation through BERT Language Model Finetuning for Aspect-Target Sentiment Classification" article by Alexander Rietzler et al. https://arxiv.org/pdf/1908.11860.pdf. 

To produce the contextualized embeddings, we use the pre-trained model **BERT base uncased**. We chose the base one as it's light in memory, and needs less time to fine-tune.

### Pre-processing :

We don't compute any feature on top of the data.

The raw input data for the classifier is composed by a Sentence, an Aspect term and an Aspect category. 

We first tokenize the input. The data given to the tokenizer is the following string "sentence. [SEP] aspect term [SEP] Aspect category". Indeed, we want to take advantage from the attention mechanism to produce embeddings conditionned by the aspact category and the aspect term. The chosen order is the one with the best performances.

Finally, the input for the transformer is : "[CLS] token_sentence [SEP] token_aspect_term [SEP] token_aspect_category [SEP]".

The length of each data point is quite small, ie nerver exceeding 200 tokens. Thus, there is no need to truncate it before giving it to the transformer model.

The output is a contextalized embedding for each token.
  
We add a column to convert the labels into numerical data : negative = 0, neutral = 1, positive = 2.

### Classifier

  The classification task is performed on top of the [CLS] token hidden representation - as in the self-supervised learning of BERT  used for sequence-pair classification -. We take advantage of a shallow neural network with only one layer outputting the probability over the 3 classes : negative, neutral, positive, with a softmax at the end. The argmax is taken to perform the prediction.
  
  The optimizer chosen is the well known Adam with a quite small learning rate : 0.00001.
 
 
 ## 3. Accurary on the dev set
 
 In average, we get an accuracy of 85% on the dev set (going from 84.2% to 87.5% during on our multiple runs).
  
