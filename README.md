
Multilingual Knowledge-grounded Dialogue Systems
-------------------------------------------------------------------------------------------------------------------------------------------------------

The implementation of the master's thesis work [Can Wizards be Polyglots: Towards a Multilingual Knowledge-grounded Dialogue System](http://urn.kb.se/resolve?urn=urn:nbn:se:uu:diva-477541). 

This thesis invesitgated if and how can we develop a multilingual knowledge-ground dialogue system without in-task data by leveraging various transfer learning techniques, given the lack of multilingual non-task-oriented data. The goal was to explore if cross-task transfer (multilingual question answering to dialogue) or cross-lingual transfer (English to multilingual) could be useful for a non-task-oriented, knowledge-aware dialogie model in a multilingual setting. I focused on five typologically diverse languages, namely Arabic, Bengali, Finnish, Japanese, and Korean, of which well-performing models could generalize to the languages that are part of the language family as the target languages, hence widening the accessibility of the systems to speakers of various languages.

The script provided in this repo includes the necessary steps to reproduce the result presented in the master's thesis. Two approaches were proposed and implemented:

1. Multilingual Retrieval-Augmented Dialogue Model (xRAD)
2. Multilingual Generative Dialogue Model (xGenD)

xRAD was adopted from a pre-trained multilingual question answering (QA) system ([CORA, Asai et al., 2021](https://arxiv.org/abs/2107.11976)) and comprises a multilingual neural retriever (mDPR) and a multilingual generation model (mGEN) with modification in the model arhitecture to add in dialogue context, followed by multi-task training to perform transfer learning.

xGenD took advantage of an existing English dialogue model (BlenderBot 400M Distilled) and performed a zero-shot cross-lingual transfer by training sequentially on English dialogue and multilingual QA datasets.

## Contents
1. [Data and Pre-trained Models](https://github.com/evelynkyl/xRAD_multilingual_dialog_systems#Data-and-Pre-trained-Models)
2. [Pre-requisite and Installation](https://github.com/evelynkyl/xRAD_multilingual_dialog_systems#Pre-requisite-and-Installation)
3. [Training](https://github.com/evelynkyl/xRAD_multilingual_dialog_systems#Training)
4. [Evaluation](https://github.com/evelynkyl/xRAD_multilingual_dialog_systems#Evaluation)
5. [Citations and Contact](https://github.com/evelynkyl/xRAD_multilingual_dialog_systems#Citations-and-Contact)

## Data and Pre-trained Models
1. Download the datasets
Unzip the dataset and then place it in a suitable location

### Data
1. mGEN training data
```bash
mkdir data
cd data
wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_dev_full_v1_1.jsonl
wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_test_full_q_only_v1_1.jsonl
cd ..
```
2. LFQA data
```bash
wget https://huggingface.co/datasets/vblagoje/lfqa/blob/main/train.json
wget https://huggingface.co/datasets/vblagoje/lfqa/blob/main/validation.json
wget https://huggingface.co/datasets/vblagoje/lfqa/blob/main/test.json
```

### Trained models
```bash
mkdir models
wget https://nlp.cs.washington.edu/xorqa/cora/models/all_w100.tsv
wget https://nlp.cs.washington.edu/xorqa/cora/models/mGEN_model.zip
wget https://nlp.cs.washington.edu/xorqa/cora/models/mDPR_biencoder_best.cpt
```
### mDPR dense embeddings to create a FAISS index
```
unzip mGEN_model.zip
mkdir embeddings
cd embeddings
for i in 0 1 2 3 4 5 6 7;
do 
  wget https://nlp.cs.washington.edu/xorqa/cora/models/wikipedia_split/wiki_emb_en_$i 
done
for i in 0 1 2 3 4 5 6 7;
do 
  wget https://nlp.cs.washington.edu/xorqa/cora/models/wikipedia_split/wiki_emb_others_$i  
done
cd ../..
```


## Pre-requisite and Installation
To run the script, 

### Technical requirements
Please note that running the script of this project is quite computationally expensive. You should have at least 256GB of RAM alloocated for pre-training and a CUDA enabled GPU with 96GB of memory.

For the training of the xGenD model, 64GB of RAM and a smaller GPU (24GB of memory) should be sufficient.

### Dependencies
The following applications and libraries needs to be installed in order to run the application.

1. Miniconda or Anaconda with Python3
2. [ParlAI](http://parl.ai)
3. [PyTorch](https://pytorch.org/get-started/locally/)
4. [Mephisto](https://github.com/facebookresearch/Mephisto) for human evaluation
5. CUDA enabled GPU

### Installation
1. Download this repo
```bash
git clone https://github.com/evelynkyl/xRAD_multilingual_dialog_systems.git
```
2. Download the pre-trained model


## Baseline
In our paper, we have tested several baselines such as Translate-test or multilingual baselines. The codes for machine translations or BM 25-based retrievers are at baselines. To run the baselines, you may need to download code and mdoels from the XOR QA repository.

## Examples
Evaluate a model on the test set of the Blended Skill Talk task:
```bash
parlai eval_model -m ir_baseline -t blendedskilltalk -dt test
```
Train a single layer transformer on PersonaChat (requires pytorch and torchtext).
Detail: embedding size 300, 4 attention heads,  2 epochs using batchsize 64, word vectors are initialized with fasttext and the other elements of the batch are used as negative during training.
```bash
parlai train_model -t personachat -m transformer/ranker -mf /tmp/model_tr6 --n-layers 1 --embedding-size 300 --ffn-size 600 --n-heads 4 --num-epochs 2 -veps 0.25 -bs 64 -lr 0.001 --dropout 0.1 --embedding-type fasttext_cc --candidates batch
```

## Acknowledgment 
I am grateful to the authors of CORA and DPR for providing reproducible training scripts and ParlAI for open-sourcing their models and frameworks. I would like to thank my supervisor at Johan Sjons at Uppsala University for his invaluable feedback and insights. I would also like to thank the volunteering participants for providing assessments of the dialogue systems and their invaluable feedback, which is immensely helpful for analyzing and interpreting the performance and quality of the conversational AI systems.


## Citation and Contact
For more details of the thesis, you can read the thesis [here](http://urn.kb.se/resolve?urn=urn:nbn:se:uu:diva-477541).

If you find this codebase useful or use in your work, please cite the thesis.

```
@mastersthesis{Liu1671559,
   author = {Liu, Evelyn Kai Yan},
   institution = {Uppsala University, Department of Linguistics and Philology},
   pages = {125},
   school = {Uppsala University, Department of Linguistics and Philology},
   title = {Can Wizards be Polyglots: Towards a Multilingual Knowledge-grounded Dialogue System },
   keywords = {Knowledge-grounded dialogue, Dialogue systems, Generative question answering, Multilingual question answering, Multilingual dialogue systems, Transfer learning, Multi-task learning, Sequential training, Conversational AI, Natural Language Processing (NLP), Deep learning, Machine learning},
   year = {2022}
}
```

If you have any questions regarding the code or the thesis, please don't hesitate to post on the issue page of this repo or contact [evelyn.kyliu.uu@gmail.com](mailto:evelyn.kyliu.uu@gmail.com).
