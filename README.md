# Hierarchical Gaussian Model with Maximum A Posteriori estimation (HGM-MAP) 

HGM-MAP was proposed to learn word and context representations simultaneously. Please cite the following paper (to appear) if you use our source code.
```
@inproceedings{aaai-feng:18,
	author = {Jiangtao Feng and Xiaoqing Zheng},
	title = {Geometric Relationship between Word and Context Representations},
	booktitle = {AAAI Conference on Artificial Intelligence},
	year = {2018}
}
```

We would thank the authors of word2vec [1] and GloVe [2] for sharing their source codes.

## Instructions
We here give an implementation modified from the source code of word2vec (https://code.google.com/p/word2vec/).
1. Compile the source.code with `clang HGM.c HGM -lm -lpthread`
2. Run `./HGM` with following arguments, e.g. `./HGM -train wikipedia.txt -context context.txt -word word.txt -weight weight.txt -d 300 -alpha 0.025 -beta 1e-3 -window 5 -sample 1e-5 -negative 5 -iter 3 -threads 8`:
	- `-train`: unannotated textual corpus as training data
	- `-context`: file to store word embeddings used for generating context representations
	- `-word`: file to store word embedding used for computing word-context similarity
	- `-weight`: file to store weight and bias in context generation model  
	- `-d`: dimensionality of word embeddings [`default 300`]
	- `-alpha`: learning rate [default `0.025`]
	- `-beta`: regularization rate [default `1e-3`]
	- `-window`: window size [default `5`]
	- `-sample`: subsampling rate [default `1e-5`]
	- `-negative`: number of negative samples [default `5`]
	- `-iter`: iteration times [default `3`]
	- `-threads`: thread number to run the code [default `8`]

## References
1. Mikolov, T.; Chen, K.; Corrado, G.; and Dean, J. 2013. Efficient estimation of word representations in vector space. CoRR abs/1301.3781.
2. Pennington, J.; Socher, R.; and Manning, C. D. 2014. Glove: Global vectors for word representation. In Proceedings of the International Conference on Empirical Methods in Natural Language Processing (EMNLP’14), volume 14, 1532–1543.
