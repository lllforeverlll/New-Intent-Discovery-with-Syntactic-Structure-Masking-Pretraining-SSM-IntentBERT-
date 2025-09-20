import json





def duqu():

    datas=[]
    dataFilePath='cixing.json'
    with open(dataFilePath, 'r') as json_file:
        dataList=(json.load(json_file))
        
        
    print(dataList)
    
def main():
    duqu()
        
if __name__ == "__main__":
    main()
    exit(0)