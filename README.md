# Learning by Semantic Similarity Makes Abstractive Summarization Better

### Author's note
The initial version of the manuscript includes a model design (semsim), experimental results on our model, BART model, and the reference summaries (both automatic evaluation metric and human evaluation metric), and discussions on the results. After we archived the manuscript, we found that our model has flaws in its implementation and design. 

[The final version of the manuscript](https://arxiv.org/pdf/2002.07767v2.pdf) is from the rest of the initial paper; we included our findings on the benchmark dataset, BART generated results and human evaluations, and we excluded our model semsim.


## Folder description
```
/--|datasets/
   |results/
   |README.md
   |model.jpg
```
   
*  `/datasets`  : Our version of the pre-processed CNN/DM dataset and the pre-processing code. Modified from [PGN by See et al.](https://github.com/abisee/cnn-dailymail) following instructions of [BART (issue #1391)](https://github.com/pytorch/fairseq/issues/1391)
*  `/results` : We provide summarization results for the CNN/DM dataset and the reduced dataset (n=1000). Folder contains generated summaries of BART and SemSim and reference summaries (not tokenized). 
