# NLP_assignment

## 1. Name of the students

Pierre Boscherel, Alexis-Raja Brachet, Laurane Gorgues, Anouar Oussalah

## 2. Description of the classifier

The overall architecture and methodology has been adapted from the "Adapt or Get Left Behind:Domain Adaptation through BERT Language Model Finetuning for Aspect-Target Sentiment Classification" article by Alexander Rietzler et al. https://arxiv.org/pdf/1908.11860.pdf. 

To produce the contextualized embeddings, we use the pre-trained model **Roberta base**. We chose the base one as it's light in memory.

### Pre-processing :

We don't compute any hand-features on top of the data.

The raw input data for the classifier is composed by a Sentence, an Aspect term and an Aspect category. 

We first tokenize the input. The data given to the tokenizer is the following string "sentence [SEP] aspect term [SEP] Aspect category". Indeed, we want to take advantage from the attention mechanism to produce embeddings conditionned by the aspect category and the aspect term. The chosen order is the one with the best performances.

Finally, the input for the transformer is : "[CLS] token_sentence [SEP] token_aspect_term [SEP] token_aspect_category [SEP]".

The length of each data point is quite small, ie nerver exceeds 200 tokens. Thus, there is no need to truncate it before giving it to the transformer model.

The output of the transformer is a contextalized embedding for each token.
  
We add a column to convert the labels into numerical data : negative = 0, neutral = 1, positive = 2.

### Classifier

  The classification task is performed on top of the [CLS] token hidden representation - as in the self-supervised learning of BERT used for sequence-pair classification -. We take advantage of a shallow neural network with only one layer outputting the probability over the 3 classes : negative, neutral, positive, with a softmax at the end. The argmax is taken to perform the prediction.
  
  The optimizer chosen is the well known Adam with a quite small learning rate : 0.00001. We choose a batch size of 32 and 20 epochs. As the data labels are unbalanced, we thought about wisely build each batch to prevent to have batch with only the positive label, but it gave poorer results.
 
 
 ## 3. Accurary on the dev set
 
 The best run we got produced an average accuracy of 88.24% on the dev set with a standard deviation of 0.81 ([87.77 89.63 88.56 87.23 88.03]).
 For some ponctual runs, the model predicted positive for all the data points of the dev set, getting a 70% accuracy. We investigated the reason why but we haven't found it during our experimentations.
  
