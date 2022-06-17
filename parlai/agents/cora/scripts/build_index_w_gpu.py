import faiss
import numpy as np
import pickle
import os
import logging
import torch
logging.basicConfig(filename='logs.log',
                    filemode='w',
                    level=logging.INFO)
logger=logging.getLogger()
logger.setLevel(logging.DEBUG)


def get_input_file(embs_name='wiki_emb', index_dir = "/root/cora/embeddings/"):  
    en_num_parts = len([f for f in os.listdir(index_dir)
                        if 'sample' not in f and 'en' in f])
    
    others_num_parts = len([f for f in os.listdir(index_dir)
                        if 'sample' not in f and 'others' in f])
    en_input_files = [
        os.path.join(index_dir, f'{embs_name}_en_{i}') for i in range(en_num_parts)
    ]
    others_input_files = [
        os.path.join(index_dir, f'{embs_name}_others_{i}') for i in range(others_num_parts)
    ]
 #   print("type of en_input_files is ", type(en_input_files))
    input_files = en_input_files + others_input_files
  #  print("this is the input files", input_files)
    return input_files
    
    
def iterate_encoded_files(input_files):
    tensors = []
    for i, file in enumerate(input_files):
        print('Reading file ', file)
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                db_id, doc_vector = doc
            #    yield doc_vector    
             #   tensor_doc_vector = torch.from_numpy(doc_vector)          
                tensors.append(doc_vector) #tensor_doc_vector
    return tensors
                
def train_then_add():
    print("Configuring the faiss index...")
    index = faiss.index_factory(128, "OPQ32_128,IVF1048576_HNSW32,PQ32") # 262144
    index_ivf = faiss.extract_index_ivf(index)
    clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
    index_ivf.clustering_index = clustering_index
    print("Configuration done")
    print("Reading the embeddings")
    read_emd = get_input_file('wiki_emb', "/root/cora/embeddings/")
    print("Done reading the embeddings")
    print("Started loading the embeddings")
    load_emb = iterate_encoded_files(read_emd)
    print("Loading is done!")
    print("this is load_emb: ", load_emb)
   # print("this is torch cat load : ", [torch.cat(load_emb)])
    cat_emb = np.concatenate(load_emb)

    print("Started building the compressed faiss index...")
    index.train([cat_emb])
   # index.train([torch.cat(load_emb)])
    print("the faiss index is complete!")
    print("adding faiss index!")
    index.add([cat_emb])
  #  index.add([torch.cat(load_emb)])
    print("adding faiss index is complete!")
    print("saving the completed index!")
    index.serialize("/root/cora/emb_index/IVF262144_HNSW32__PQ64.index")
    print("the index has been saved in '/root/cora/emb_index/IVF262144_HNSW32__PQ64.index'! ")

  #  all_docs = []
  #      db_id, doc_vector = item
  #      all_docs.append((db_id, doc_vector)) 
   # return doc_vectors

#def save_index(idx, save_dir="/root/cora/emb_index/IVF262144_HNSW32__PQ64.index"):
 #   return idx.serialize(save_dir)
 
if __name__ == "__main__":
    train_then_add()
    
    
#def index_data():
#    all_docs = []
  #  for i, item in enumerate(self.iterate_encoded_files(input_files)):
 #       logging.info(f'Loading in file {item}')
  #      db_id, doc_vector = item
  #      all_docs.append((db_id, doc_vector))
  #  self.index.index_data(all_docs)
# logging.info('Data indexing comple")
                     
