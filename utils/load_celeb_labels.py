import pandas as pd
import numpy as np
dir_anno = "/home/xtreme/runs/data/celeba/anno/"

def get_annotation(fnmtxt, verbose=True):
    if verbose:
        print("_"*70)
        print(fnmtxt)
    
    rfile = open( dir_anno + fnmtxt , 'r' ) 
    texts = rfile.read().split("\n") 
    print(type(texts))
    print(len(texts))
    rfile.close()

    columns = np.array(texts[1].split(" "))
    columns = columns[columns != ""]
    df = []
    for txt in texts[2:]:
        txt = np.array(txt.split(" "))
        txt = txt[txt!= ""]
    
        df.append(txt)
        
    df = pd.DataFrame(df)

    if df.shape[1] == len(columns) + 1:
        columns = ["image_id"]+ list(columns)
    df.columns = columns   
    df = df.dropna()
    if verbose:
        print(" Total number of annotations {}\n".format(df.shape))
        print(df.head())
    ## cast to integer
    for nm in df.columns:
        if nm != "image_id":
            df[nm] = pd.to_numeric(df[nm],downcast="integer")
    return(df)
print(get_annotation("list_attr_celeba.txt"))