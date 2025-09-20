import json
import stanza




def hebing():
    datas=[]


    dataFilePath='dataset.json'
    with open(dataFilePath, 'r') as json_file:
        dataList1=(json.load(json_file))
        

    dataFilePath='MCIDpos.json'
    with open(dataFilePath, 'r') as json_file:
        dataList2=(json.load(json_file))
    b1=dataList1["stackoverflow"]
    t=0

    c1=b1["STACKOVERFLOW"]
    b2=dataList2
    for i in range(0,len(c1)):
    # for i in range(0,70):
        print("第",i)
        c2=b2[i+t]['tword']
        q=0
        n=0

        y=0
        for j in range (0,len(c2)):
            p=len(c2[j])
            y=y+p
            if c2[j]=="-" or c2[j]=="," or c2[j]=="?" or c2[j]=="'s" or c2[j]=="."or c2[j]=="+"or c2[j]=="("or c2[j]==")" or c2[j]==":" or c2[j]=="["or c2[j]=="]":
                if c2[j]=="-"or c2[j]=="+"or c2[j]=="/"or c2[j]=="("or c2[j]=="[" :
                    q=q+2
                else: 
                    q=q+1
            print("y",y+len(c2)-1,"句子长度",len((c1[i][0])),y+len(c2)-1-q)
            

            
        if y+len(c2)-1 ==len((c1[i][0]))or y+len(c2)+2-q ==len((c1[i][0])) or y+len(c2)-1-q ==len((c1[i][0]))or y+len(c2)-3-q ==len((c1[i][0])) :

            print("true")
            datas.append(b2[i+t]['tword'])
            
        else:
            print("false")
            # r1=b2[i+t]['tupos']
            r1=b2[i+t]['tword']
            t=t+1
            c3=b2[i+t]['tword']
            r2=b2[i+t]['tword']
            for j in range (0,len(c3)):
                p=len(c3[j])
                y=y+p
                if c3[j]=="-" or c3[j]=="," or c3[j]=="?" or c3[j]=="'s" or c3[j]=="."or c3[j]=="+"or c3[j]=="("or c2[j]==")" or c3[j]==":" or c3[j]=="["or c3[j]=="]":
                    if c3[j]=="-"or c3[j]=="+"or c3[j]=="/"or c3[j]=="("or c3[j]=="[" :
                        n=n+2
                    else: 
                        n=n+1
                if y+len(c2)+len(c3)-1 ==len((c1[i][0])) or y+len(c2)+len(c3)-1-q or y+len(c2)+len(c3)-2 ==len((c1[i][0])):
                    b4=r1+r2
                    datas.append(b4)
            
            
            

    file_name='tupos.json'
    with open(file_name,'w') as file_object:
        json.dump(datas,file_object)
    # c2=b2['tword']
    # c3=b3['tword']
    # print((c1[10][0]))
    # print(len(c2[0]))

def makenv():
    
    
    dataFilePath='dataset.json'
    with open(dataFilePath, 'r') as json_file:
        dataList1=(json.load(json_file))
        
      
    b1=dataList1["stackoverflow"]
    c1=b1["STACKOVERFLOW"]
    data=c1
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')
    datas=[]
    key = ['tword', 'tupos']
    a = dict([(k, []) for k in key])
    file_name = 'MCIDpos.json'
    
    for i in range (0,len(data)):
        # print(i)
        doc = nlp(data[i][0])
        
        
        

        # print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}\tupos: {word.upos}\txpos: {word.xpos}' for sent in doc.sentences for word in sent.words], sep='\n')
        # dict={'txpos':word.xp,}
        

        if len(doc.sentences)>1:
            for sent in doc.sentences :
                for word in sent.words:

                    a["tword"].append(word.text)

                    a["tupos"].append(word.upos)

            datas.append(a)
            a = dict([(k, []) for k in key])
        else:    
            for sent in doc.sentences :
                for word in sent.words:

                    a["tword"].append(word.text)

                    a["tupos"].append(word.upos)

                datas.append(a)
                a = dict([(k, []) for k in key])
        

    with open(file_name,'w') as file_object:
        json.dump(datas,file_object)    
    
    
    
    
    
    
def main():
    makenv()
        
if __name__ == "__main__":
    main()
    exit(0)