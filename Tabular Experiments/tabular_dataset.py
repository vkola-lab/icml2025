import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Subset, TensorDataset
from collections import Counter

# this .py file is used to load the datasets for the tabular data experiments and taken from the Greedy paper's code [1].
# [1] Covert, Ian Connick, et al. "Learning to maximize mutual information for dynamic feature selection." International Conference on Machine Learning. PMLR, 2023.
# this is the url for the datasets in the dynamic-selection repo
GITHUB_URL = 'https://raw.githubusercontent.com/iancovert/dynamic-selection/main/datasets/'
        

def count_labels(subset):
    labels = [subset.dataset.tensors[1][i].item() for i in subset.indices]  # Extract labels from subset
    return Counter(labels)

def data_split(dataset, val_portion=0.2, test_portion=0.2, random_state=0, tree_return=False, if_metabric=False, if_ckd=False):
    '''
    Split dataset into train, val, test.
    
    Args:
      dataset: PyTorch dataset object.
      val_portion: percentage of samples for validation.
      test_portion: percentage of samples for testing.
      random_state: random seed.
    '''
    # Shuffle sample indices.
    rng = np.random.default_rng(random_state)
    inds = np.arange(len(dataset))
    rng.shuffle(inds)

    # Assign indices to splits.
    n_val = int(val_portion * len(dataset))
    n_test = int(test_portion * len(dataset))
    test_inds = inds[:n_test]
    
    val_inds = inds[n_test:(n_test + n_val)]
    train_inds = inds[(n_test+ n_val):] #  + n_val
    

    
    #print(train_inds)
    # Create split datasets.
    test_dataset = Subset(dataset, test_inds)
    val_dataset = Subset(dataset, val_inds)
    train_dataset = Subset(dataset, train_inds)

    
    if if_metabric==True:
        train_x = dataset.tensors[0][train_inds,:]
        #train_y = dataset.tensors[1][train_inds]
        train_y = dataset.tensors[2][train_inds]
        shap_values_metabric_train= np.load('metabric_weighted_sampling_shap_custom_shap_train_nsamples4096.npy')   # aids_v0_tree_depth2_lr005_weighted_now2 #aids_v0_tree_depth2_lr005_weighted
        shap_values_metabric_train=torch.tensor(shap_values_metabric_train)
        train_dataset = TensorDataset(train_x,  shap_values_metabric_train, train_y)

    if if_ckd==True:
        train_x = dataset.tensors[0][train_inds,:]
        #train_y = dataset.tensors[1][train_inds]
        train_y = dataset.tensors[2][train_inds]
        shap_values_metabric_train= np.load('kidney_sampling_shap_custom_shap_train_nsamples2048_correct_second_calc.npy')   # aids_v0_tree_depth2_lr005_weighted_now2 #aids_v0_tree_depth2_lr005_weighted
        shap_values_metabric_train=torch.tensor(shap_values_metabric_train)
        train_dataset = TensorDataset(train_x,  shap_values_metabric_train, train_y)

        
    if tree_return==True:
        train_x = dataset.tensors[0][train_inds,:]
        #train_x_shap = dataset.tensors[1][train_inds]
        #train_y = dataset.tensors[1][train_inds]
        train_y = dataset.tensors[2][train_inds]

        val_x = dataset.tensors[0][val_inds,:]
        #val_y = dataset.tensors[1][val_inds]
        val_y = dataset.tensors[2][val_inds]

        test_x = dataset.tensors[0][test_inds,:]
        #test_y = dataset.tensors[1][test_inds]
        test_y = dataset.tensors[2][test_inds]

        return train_x,  train_y, val_x, val_y, test_x, test_y # train_x_shap is required for AACO
    else:
        return train_dataset, val_dataset, test_dataset#, val_inds



def load_cir(features=None):
    # Load data.
    import pandas as pd
    df = pd.read_csv('/projectnb/vkolagrp/ketanss/dynamic-selection/experiments/tabular/cir_pre.csv')
    # Set features.
    df['Outcome'] = df['Status']
    df.drop(columns=['Status'], inplace=True)
    if features is None:
        features = np.array([f for f in df.columns if f not in ['Outcome']])
    else:
        assert 'Outcome' not in features
        if isinstance(features, list):
            features = np.array(features)
    # Extract x, y.
    x = np.array(df.drop(['Outcome'], axis=1)[features]).astype('float32')
    y = np.array(df['Outcome']).astype('int64')
    # Create dataset object.
       
    

    
    shap_values = np.load('cirr_oracle_shap.npy')   # aids_v0_tree_depth2_lr005_weighted_now2 #aids_v0_tree_depth2_lr005_weighted
    shap_values=torch.tensor(shap_values)
    dataset = TensorDataset(torch.from_numpy(x),  shap_values, torch.from_numpy(y)) # shap_values_spam
    dataset.features = features
    dataset.input_size = x.shape[1]
    dataset.output_size = len(np.unique(y))
    return dataset
    
def load_aids(features=None):
    # Load data.
    import pandas as pd
    df = pd.read_csv('/projectnb/vkolagrp/ketanss/dynamic-selection/experiments/tabular/aids.csv')

    if features is None:
        features = np.array([f for f in df.columns if f not in ['Outcome']])
    else:
        assert 'Outcome' not in features
        if isinstance(features, list):
            features = np.array(features)
    # Extract x, y.

    # aids_v0_tree_depth2_lr005_weighted
    x = np.array(df.drop(['Outcome'], axis=1)[features]).astype('float32')
    y = np.array(df['Outcome']).astype('int64')
    # Create dataset object.

    shap_values = np.load('aids_oracle_shap.npy')  
    shap_values=torch.tensor(shap_values)

    dataset = TensorDataset(torch.from_numpy(x),  shap_values, torch.from_numpy(y)) #shap_values_spam,
    dataset.features = features
    dataset.input_size = x.shape[1]
    dataset.output_size = len(np.unique(y))
    return dataset
    


def load_metabric(features=None):
    pam50_mapping = {
        'LumA': 0,
        'LumB': 1,
        'Her2': 2,
        'claudin-low': 3,
        'Basal': 4,
        'Normal': 5
    }
    data = pd.read_csv("/projectnb/vkolagrp/datasets/Metabric/METABRIC_RNA_classification.csv")
    data['pam50_+_claudin-low_subtype'] = data['pam50_+_claudin-low_subtype'].map(pam50_mapping)

    data = data.dropna(subset=['pam50_+_claudin-low_subtype'])

    data = data.drop(columns=['neoplasm_histologic_grade', 'cancer_type_detailed'])
    data = data.rename(columns={'pam50_+_claudin-low_subtype': 'Outcome'})

    if features is None:
        features = np.array([f for f in data.columns if f not in ['Outcome']])
    else:
        assert 'Outcome' not in features
        if isinstance(features, list):
            features = np.array(features)
    # Extract x, y.
    x = np.array(data.drop(['Outcome'], axis=1)[features]).astype('float32')
    y = np.array(data['Outcome']).astype('int64')

    shap_values = np.load('metabric_tree.npy') # metabric_surr_v37_deep | (tree_based): metabric_surr_v22_deep 
    
    # metabric_tree
    # metabric_oracle_shap
    # metabric_lime_shap
    # metabric_oracle_shap
    # metabric_sampling_shap

    shap_values=torch.tensor(shap_values)
    # Create dataset object.
    dataset = TensorDataset(torch.from_numpy(x),  shap_values, torch.from_numpy(y)) # shap_values_spam
    dataset.features = features
    dataset.input_size = x.shape[1]
    dataset.output_size = len(np.unique(y))

    
    return dataset


def load_ckd(features=None):
    df = pd.read_csv('/projectnb/vkolagrp/ketanss/dynamic-selection/experiments/tabular/ckd.csv')
    df['Outcome'] = df['Diagnosis']
    df = df.drop(columns=['PatientID','Ethnicity','DoctorInCharge','Diagnosis'])
    if features is None:
        features = np.array([f for f in df.columns if f not in ['Outcome']])
    else:
        assert 'Outcome' not in features
        if isinstance(features, list):
            features = np.array(features)
    # Extract x, y.
    x = np.array(df.drop(['Outcome'], axis=1)[features]).astype('float32')
    y = np.array(df['Outcome']).astype('int64')
    # Create dataset object.

    #good:
    #kidney_v0_tree_depth2_lr005_weighted

    #bad:
    #kidney_v0_tree_depth2_lr005_weighted_combined

    #kidney_sampling_shap
    #kidney_oracle_shap
    #kidney_v0_tree_depth2_lr005_weighted
    #ckd_invase
    shap_values_spam = np.load('kidney_oracle_shap.npy') 
    
   
    shap_values_spam=torch.tensor(shap_values_spam)
    # Create dataset object.
    dataset = TensorDataset(torch.from_numpy(x), shap_values_spam, torch.from_numpy(y)) #shap_values_spam

    #dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    dataset.features = features
    dataset.input_size = x.shape[1]
    dataset.output_size = len(np.unique(y))
    return dataset

def load_spam(features=None):
    # Load data.
    data_dir = os.path.join(GITHUB_URL, 'spam.csv')
    data = pd.read_csv(data_dir)
    
    # Set features.
    if features is None:
        features = np.array([f for f in data.columns if f not in ['Outcome']])
    else:
        assert 'Outcome' not in features
        if isinstance(features, list):
            features = np.array(features)
            
    # Extract x, y.
    
    x = np.array(data.drop(['Outcome'], axis=1)[features]).astype('float32')
    y = np.array(data['Outcome']).astype('int64')
    
    #spam_v1_tree_depth4_lr01
    shap_values_spam = np.load('spam_v1_tree_depth4_lr01.npy')  #_kernel_shap
    
   
    shap_values_spam=torch.tensor(shap_values_spam)
    # Create dataset object.
    dataset = TensorDataset(torch.from_numpy(x), shap_values_spam, torch.from_numpy(y)) #shap_values_spam

   
    
    dataset.features = features
    dataset.input_size = x.shape[1]
    dataset.output_size = len(np.unique(y))
    return dataset


# Transform registry for easy access
DATASET_FUNCTIONS = {
    'aids': load_aids,
    'ckd': load_ckd,
    'cir': load_cir,
    'metabric': load_metabric,
    'spam': load_spam,
}

def get_dataset(dataset_name, **kwargs):
    """
    Main function to get transforms for any dataset
    
    Args:
        dataset_name: Name of the dataset ('cifar10', 'cifar100', 'imagenette', 'bloodmnist')
        **kwargs: Additional arguments passed to the specific transform function
    
    Returns:
        tuple: (pretrain_loader, trainloader, valloader, testloader)
    """
    if dataset_name not in DATASET_FUNCTIONS:
        available = list(DATASET_FUNCTIONS.keys())
        raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {available}")
    
    return DATASET_FUNCTIONS[dataset_name](**kwargs)



