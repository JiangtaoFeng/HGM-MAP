# HGM

Hierarchical Gaussian Model with Maximum A Posteriori estimation (HGM-MAP) 
________________________________________________________________________________

HGM-MAP was proposed to learn word and context representations simultaneously. Please cite the following paper if you use our source code.

@inproceedings{aaai-feng:18,
  author = {Jiangtao Feng and Xiaoqing Zheng},
  title = {Geometric Relationship between Word and Context Representations},
  booktitle = {AAAI Conference on Artificial Intelligence},
  year = {2018},
}
(TO APPEAR)

We here give a implementation modified from the source code of word2vec.

Instruction:
1. Compile the source.code with "clang HGM.c HGM -lm -lpthread"
2. Run "./HGM" with following arguments, e.g. "./HGM -train wikipedia.txt -context context.txt -word word.txt -weight weight.txt -d 300 -alpha 0.025 -beta 1e-3 -window 5 -sample 1e-5 -negative 5 -iter 3 -threads 8":
	1) -train: unannotated textual corpus as training data
	2) -context: file to store word embeddings used for generating context representations
	3) -word: file to store word embedding used for computing word-context similarity
	4) -weight: file to store weight and bias in context generation model  
	5) -d: dimensionality of word embeddings [default 300]
	6) -alpha: learning rate [default 0.025]
	7) -beta: regularization rate [default 1e-3]
	8) -window: window size [default 5]
	9) -sample: subsampling rate [default 1e-5]
	10) -negative: number of negative samples [default 5]
	11) -iter: iteration times [default 3]
	12) -threads: thread number to run the code [default 8]

We would thank the authors of word2vec and GloVe for sharing their source codes.
