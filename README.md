
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
1. [Data and Pre-trained Models](https://github.com/evelynkyl/xRAD_multilingual_dialog_systems#data-and-pre-trained-models)
2. [Pre-requisite and Installation](https://github.com/evelynkyl/xRAD_multilingual_dialog_systems#pre-requisite-and-installation)
3. [Training](https://github.com/evelynkyl/xRAD_multilingual_dialog_systems#training)
4. [Baseline](https://github.com/evelynkyl/xRAD_multilingual_dialog_systems#baseline)
4. [Evaluation](https://github.com/evelynkyl/xRAD_multilingual_dialog_systems#evaluation)
5. [Citation and Contact](https://github.com/evelynkyl/xRAD_multilingual_dialog_systems#citation-and-contact)

## [Data and Pre-trained Models](#data-and-pre-trained-models)
1. Download the datasets
Unzip the datasets and then place them in a suitable location

### Data
1. mGEN training data from [XOR-TyDi QA (Asai et al., 2021)](https://nlp.cs.washington.edu/xorqa/)
```bash
mkdir data
cd data
wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_dev_full_v1_1.jsonl
wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_test_full_q_only_v1_1.jsonl
cd ..
```
2. [LFQA data](https://huggingface.co/spaces/lfqa/lfqa)
```bash
wget https://huggingface.co/datasets/vblagoje/lfqa/blob/main/train.json
wget https://huggingface.co/datasets/vblagoje/lfqa/blob/main/validation.json
wget https://huggingface.co/datasets/vblagoje/lfqa/blob/main/test.json
```

### Trained models
Download the pre-trained mDPR and mGEN models from [CORA, Asai et al., 2021](https://arxiv.org/abs/2107.11976).
```bash
mkdir models
wget https://nlp.cs.washington.edu/xorqa/cora/models/all_w100.tsv
wget https://nlp.cs.washington.edu/xorqa/cora/models/mGEN_model.zip
wget https://nlp.cs.washington.edu/xorqa/cora/models/mDPR_biencoder_best.cpt
unzip -xf all_w100.tsv
tar -xf mDPR_cpt
unzip mGEN_model.zip
```

### mDPR dense embeddings
Download the multilingual dense embeddings of mDPR for building a FAISS index.
```
unzip mGEN_model.zip
mkdir cora/embeddings
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

## [Pre-requisite and Installation](pre-requisite-and-installation)
### Technical requirements
Please note that running the script of this project is quite computationally expensive. You should have at least 256GB of RAM alloocated for pre-training and a CUDA enabled GPU with 96GB of memory (I used NVIDIA A40 with 96GB vram to train a batch size of 1).

For the training of the xGenD model, 64GB of RAM and a smaller GPU (24GB of memory) should be sufficient.

### Dependencies
The following applications and libraries need to be installed in order to run the application.

1. [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) with Python3
2. [ParlAI](http://parl.ai)
3. [PyTorch](https://pytorch.org/get-started/locally/)
4. [Transformer](https://huggingface.co/docs/transformers/index)
5. [Mephisto](https://github.com/facebookresearch/Mephisto) for human evaluation
6. [fairseq](https://github.com/facebookresearch/fairseq)
7. [FAISS](https://github.com/facebookresearch/faiss)
8. CUDA enabled GPU

### Installation
1. Download this repo
```bash
git clone https://github.com/evelynkyl/xRAD_multilingual_dialog_systems.git
```

2. Install the above dependencies


## [Training](#training)
If you have multiple GPUs or have access to SLURM, you can use the follwoing instead of `train_model.py`.
```bash
# for multi-GPUs training
python multiprocessing_train.py -mf izoo: -t blended_skill_talk
```
```bash
# for distributed training using SLURM cluster
python distributed_train.py -mf izoo: -t blended_skill_talk
```

### xRAD
1. Build a compressed FAISS index using the multilingual dense embeddings 
```bash
python /parlai/agents/cora/scripts/index_dense_embeddings.py --retriever-embedding-size 768 --embeddings-dir /cora/embeddings/ --save_index_dir /cora/emb_index_exact --embeddings-name wiki_emb --indexer-type compressed   --compressed-indexer-factory IVF4096_HNSW128,PQ128
```

2. Multi-task training on the modified model architecture
Note that both mT5 and CORA were not implemented in ParlAI, so I have ported both of them to ParlAI in order to train the model on ParlAI. They are in `parlai/agents/cora`  and `internal_parlai/agents/mgen`.

```bash
python train_model.py --model cora -t blended_skill_talk,lfqa,wizard_of_wikipedia  --multitask_weights 2,1,1 --datatype train --rag-retriever-type mdpr --rag-model-type turn --generation-model mGEN --query-model mbert --rag-turn-marginalize doc_only --n-docs 5  --optimizer adafactor --dropout 0.1 --betas 0.9,0.999 --num-epochs 2 --fp16 False --save-after-valid True --warmup_updates 500  -lr 7e-06 --gradient-clip 1.0 --adam-eps 1e-08  --weight_decay 0.005 --skip-generation True --update-freq 2  -vmm min -veps 0.25 -vme 1000 --lr-scheduler reduceonplateau --lr-scheduler-patience 1  --warmup-rate 0.0001 --dynamic-batching full --mGEN_model_parallel false --thorough True --indexer-type compressed --model-file /parlai/models/xrad/xrad --inference nucleus
```

### xGenD
1. Intermediate training on English dialogue and LFQA data
```bash
python train_model.py -mf zoo:blender/blender_400Mdistill/model --model transformer/generator -t lfqa,wizard_of_wikipedia  --multitask-weights 1,2 --datatype train --dynamic-batching full --batchsize 1 --fp16 True --gradient-clip 0.4  --dict_file zoo:blender/blender_400Mdistill/model.dict --dict-lower True --dict-tokenizer bytelevelbpe --dropout 0.1 --variant prelayernorm  --embedding-size 1280  --ffn-size 5120 --n-decoder_layers 12  --n-encoder_layers 2 --n-heads 32  --n-layers 8 --n-positions 128 --weight_decay 0.005 --label-truncate 128 --lr-scheduler reduceonplateau --lr-scheduler-patience 3 --optimizer adafactor --dropout 0.1 --num-epochs 2 --truncate 512 -vmm min -veps 0.25 -vme 1000 -vmt ppl -vp 5 --save-after-valid True --warmup_updates 500  -lr 7e-06 --skip-generation True  --update-freq 2  --dict-file parlai/data/lfqa/lfqa.dict --dict-file parlai/data/wizard_of_wikipedia/wizard_of_wikipedia.dict --model-file /parlai/models/400M_genmdm_seq_exp1/400M_genmdm_seq_exp1
```

2. Second fine-tuning step on multilingual QA data
```bash
python train_model.py -mf izoo:models/400M_genmdm_seq_exp1/400M_genmdm_seq_exp1 -t xorqa,tydiqa --multitask-weights stochastic --datatype train  --inference nucleus --fp16 True --fp16-impl  mem_efficient --dynamic-batching full --batchsize 1 --gradient-clip 0.4 --dropout 0.1  -lr 7e-06  --weight_decay 0.005 --label-truncate 128 --lr-scheduler reduceonplateau --lr-scheduler-patience 3 --embedding-size 1280  --ffn-size 5120 --n-decoder_layers 12  --n-encoder_layers 2 --n-heads 32  --n-layers 8 --n-positions 128 --attention-dropout 0.0 --dict-lower True --dict-tokenizer bytelevelbpe --warmup_updates 50 -vmm min -veps 0.25 -vme 1000 -vmt ppl -vp 5 --save-after-valid True --skip-generation True --update-freq 2  --model-file /parlai/models/genmdm_400M/genmdm_400M
```

## [Evaluation](#evaluation)
### Intrinsic evaluation
It is important to note that quantative evaluation was only performed for English dialogue, given the unavailability of multilingual dialogue datasets for all five of the target languages. It is used as an estimation to the performance of the models; however, since automated metrics are still not quite reliable for evaluating open dialogue tasks, human evaluation is key. In this case, perplexity was used as the automated metric since it was reported to have stronger correlation to human judgment (Ad et al, 2021).

**xRAD**
```bash
python eval.py -mf izoo:models/xrad/xrad -t blended_skill_talk --datatype test --inference nucleus --metrics ppl
python eval.py -mf izoo:models/xrad/xrad -t wizard_of_wikipedia --datatype test --inference nucleus --metrics ppl
```

**xGenD**
```bash
python eval.py -mf izoo:models/genmdm_400M/genmdm_400M -t blended_skill_talk --datatype test --inference nucleus  --metrics ppl
python eval.py -mf izoo:models/genmdm_400M/genmdm_400M -t wizard_of_wikipedia --datatype test --inference nucleus  --metrics ppl
```

**Baseline**
To compare the two models, I have set up a Translate-test baseline, which adds a machine translation component to a pre-trained knowledge-aware open dialogue model ([BlenderBot 400M Distilled, Roller et al., 2021](https://parl.ai/projects/recipes/)).

```bash
# on English data
python eval_model.py -mf zoo:blender/blender_400Mdistill/model -t blended_skill_talk -dt test --inference nucleus --metrics ppl

python eval_model.py -mf zoo:blender/blender_400Mdistill/model -t wizard_of_wikipedia -dt test --inference nucleus --metrics ppl
```

### [Human evaluation](#human-eval)
To launch human evaluation, we will use the Mephisto framework. For more information about how to use the framework, see their [documentation](https://mephisto.ai). For each evaluation task, a configuration yaml file is required to set the parameters. Examples can be found in folder `hydra_configs/conf` . The original scripts from Mephisto were for comparing between two models, however I modified the backend and frontend so that I can compare three models (baseline, xRAD, xGenD) instead. 

#### Single model evaluation
This evaluation lets a human participant chat with the model for seven turns, and at the end of each turn the participant can rate the model response on the aspects of grammaticality (fluency), relevance, interestingness, engagingness, factual accuracy,and knowledgeableness on a scale of five. At the end of the conversation, the participant will also have the chance to give an overall rating of the model on the same conversaional qualities.

##### Sample interface


##### Run

```bash
# for local test
python /parlai/crowdsouring/tasks/model_chat/run.py conf=example_fi

# for internal sharing
python run.py mephisto/provider=mock mephisto/architect=heroku conf=example_fi

# for running live
python run.py mephisto.provider.requester_name=my_mturk_user mephisto/architect=heroku
```

#### Model preference evaluation
##### Sample interface



##### Run
```bash
# for local test
python /parlai/crowdsouring/tasks/my_pw_per_turn/run.py conf=example_fi
```


### Self-chat evaluation
This will let two pre-trained models chat with each other.

1. xGenD_full with xGenD_noInter (xGenD without the intermediate training on dialogue data)
```bash
python self_chat.py -mf izoo:models/genmdm_400M/genmdm_400M --partner_model_file izoo:models/GenMDM-run1/GenMDM-run1 -t blended_skill_talk --datatype valid --inference nucleus --num-self-chats 6 --display-examples True
```

2.xGenD_full with xGenD_noTarget (xGenD without the second fine-tuning step on the target task)
```bash
python self_chat.py -mf izoo:models/genmdm_400M/genmdm_400M --partner_model_file izoo:models/GenMDM-run2/genmdm-exp2-model -t blended_skill_talk --datatype valid --inference nucleus --num-self-chats 6 --display-examples True
```


## Acknowledgment
I am grateful to the authors of CORA, DPR, XORQA, TyDi QA, and LFQA for providing reproducible training scripts and datasets, as well as ParlAI and Mephisto for open-sourcing their models and frameworks. I would like to thank my supervisor at Johan Sjons at Uppsala University for his invaluable feedback and insights. I would also like to thank the volunteering participants for providing assessments of the dialogue systems and their invaluable feedback, which is immensely helpful for analyzing and interpreting the performance and quality of the conversational AI systems.


## [Citation and Contact](#citation-and-contact)
You can read the thesis [here](http://urn.kb.se/resolve?urn=urn:nbn:se:uu:diva-477541).

If you find this codebase useful or use it in your work, please cite this thesis.

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
