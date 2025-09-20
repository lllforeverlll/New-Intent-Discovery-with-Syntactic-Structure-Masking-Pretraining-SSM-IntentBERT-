import json

import numpy as np

def tongji():
    dataFilePath='cixing.json'
    with open(dataFilePath, 'r') as json_file:
        dataList=(json.load(json_file))
    b=np.array(dataList)
    # b=b[:,6]
    print(b)
    poslist=["ADJ","ADP","PUNCT","ADV","AUX",
             "SYM","INTJ","CCONJ","X","NOUN",
             "DET","PROPN","NUM","VERB","PART",
             "PRON","SCONJ"]
    for i in range(1,18):
        n = np.sum(b == i)

        print("数据集中",poslist[i-1],"数量为",n)
        n= 0
        
    h=b.shape[0]
    num = np.sum(b > 0)-h
    print("总词数为",num,"行数为",h)
    
    
    
    
    
    
    
    
def main():
    tongji()
        
if __name__ == "__main__":
    main()
    exit(0)