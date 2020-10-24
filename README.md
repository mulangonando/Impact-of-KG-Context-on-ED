# Impact-of-KG-Context-on-ED
This repo contains the code and instructions for our paper : "Evaluating the Impact of Knowledge Graph Context on Entity Disambiguation Models", CIKM, 2020

Find the link to the paper here : https://arxiv.org/pdf/2008.05190.pdf

## Wikipedia Experiments (CONLL-AIDA Dataset)
1. Obtain the DCA (Dynamic Context Augmentation) model code from this repo :
   https://github.com/YoungXiyuan/DCA

2. Follow the instructions for running the repos from their github
3. Replace their entity context file with our KG Entity context file under the following folder in this repo:
   DCA/ALL_TRIPLES_ent2desc.json
4. Train till convergence : We got our results at epoch 295 


## Wikipedia Experiments (Wikidata-Disamb & ISTEX Datasets)
Find the code under the folder "Wikidata" in this repo
1. There is code for RoBERTa and XLNet
2. Follow the instructions for the data from this repo :
   https://github.com/ContextScout/ned-graphs
   
   and palce it under the "data" folder in this repo
   
3. Our results can be found uder the : "Datasets and Results"
4. Feed the triples in the exact order in which they are retrieved from Wikidata (Will be cut off due to Sequence Length)

The folder "Datasets and Results" in the folder "Wikidata"; contains results and predictions from our runs.
