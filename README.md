# NLP_assignment

## 1. Name of the students

Pierre Boscherel, Alexis-Raja Brachet, Laurane Gorgues, Anouar Oussalah

## 2. Description of the classifier

The overall architecture and methodology has been adapted from the "Adapt or Get Left Behind:Domain Adaptation through BERT Language Model Finetuning for Aspect-Target Sentiment Classification" article by Alexander Rietzler et al. https://arxiv.org/pdf/1908.11860.pdf. 

To produce the contextualized embeddings, we use the pre-trained model **Roberta base**. We chose the base one as it's light in memory.

### Pre-processing :

We don't compute any hand-features on top of the data.

The raw input data for the classifier is composed by a Sentence, an Aspect term and an Aspect category. 

We first tokenize the input. The data given to the tokenizer is the following string "aspect term [SEP] sentence [SEP] Aspect category". Indeed, we want to take advantage from the attention mechanism to produce embeddings conditionned by the aspect category and the aspect term. The chosen order is the one with the best performances.

Finally, the input for the transformer is : "[CLS] token_aspect_term [SEP] token_sentence [SEP] token_aspect_category [SEP]".

The length of each data point is quite small, ie nerver exceeds 200 tokens. Thus, there is no need to truncate it before giving it to the transformer model.

The output of the transformer is a contextalized embedding for each token.
  
We add a column to convert the labels into numerical data : negative = 0, neutral = 1, positive = 2.

### Classifier

  The classification task is performed on top of the [CLS] token hidden representation - as in the self-supervised learning of BERT used for sequence-pair classification -. We take advantage of a shallow neural network with only one layer outputting the probability over the 3 classes : negative, neutral, positive, with a softmax at the end. The argmax is taken to perform the prediction.
  
  The optimizer chosen is the well known Adam with a quite small learning rate : 0.00001. We choose a batch size of 32 and 20 epochs after testing multiple other value which gave lower accuracies on the dev set. With a simple dataloader with shuffling this model gave very good results but not regular ones. Indeed, over 5 runs we could reach accuracies of 89% on the dev set on one given run and 70% on the next run. As the 70% runs were quite frequent, this would most likely decrease drasticaly the average accuracy over 5 runs on the test and dev set. Thus, we chose to use a Sampler in combination to the model described above: we used the WeightedRandomSampler in order to have training batches representative of the train set (as it is very unbalanced between the "positive", "negative" and "neutral" polarities). This had the effect of stabilizing the accuracy on the dev set but it also lowered it (the accuracy revolved more around 85% than 88% but we did not get anymore 70%).
 
 ## 3. Accurary on the dev set
 
 The best run we got using the sampler (the model we chose to implement in the end, after experimentation) produced an average accuracy of 85.11% on the dev set with a standard deviation of 1.29 ([87.23 85.64 83.78 85.11 83.78]).
  
