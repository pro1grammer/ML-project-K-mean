from math import e as e
import pandas as pd
import numpy as np
from math import log
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random as rd


class hierarchial:
    def __init__(self,k=5):
        self.k=k
        self.data=data
        self.clusters={new_list: [] for new_list in range(self.k)}
        self.distance_mactrix=np.zeros([self.data.shape[0],self.data.shape[0]],dtype=float)
        #self.distance_mactrix=pd.DataFrame([[100,17,21,31,23],[17,100,30,34,21],[21,30,100,28,39],[31,34,28,100,43],[23,21,39,43,100]],columns=['a','b','c','d','e'],index=['a','b','c','d','e'])
        
    def preprocess(self):
        shape=self.data.shape
        for column in self.data:
            mean=self.data.mean()[column]
            std=self.data.std()[column]
            for i in range(shape[0]):
                self.data.at[i,column]=(self.data.at[i,column]-mean)/std
        
    def cosine_distance(self,a,b):
        assert len(a) == len(b)
        ab_sum, a_sum, b_sum = 0, 0, 0
        for ai, bi in zip(a, b):
            ab_sum += ai * bi
            a_sum += ai * ai
            b_sum += bi * bi
        return 1-ab_sum / sqrt(a_sum * b_sum)
            
    def plot_cluster(self):
        fig = plt.figure(figsize = (10, 10))
        ax = plt.axes(projection ="3d")
        colors=["blue","green","red","yellow","cyan","orange"]
        i=0
        for key in self.clusters.keys():
            frame=pd.DataFrame(self.clusters[key])
            x=frame["mortality_rate"].tolist()
            y=frame["transmission_rate"].tolist()
            z=frame["incubation_period"].tolist()
            ax.scatter3D(x, y, z, color = colors[i])
            i+=1
        
        ax.set_xlabel('$Mortality Rate$', rotation=150)
        ax.set_ylabel('$Transmission Rate$')
        ax.set_zlabel(r'$Incubation Period$',  rotation=60)
        plt.title("Heirarchial Clustering with Complete Linkage")
        
        plt.show()
            
    def find_max_list(self,list1,list2,ind1,ind2):
        final_list=[0 for i in range(len(list1))]
        for i in range(len(list1)):
            if i!=ind2 and list1[i]==100:
                list1[i]=0
            if i!=ind1 and list2[i]==100:
                list2[i]=0
        
        for i in range(len(list1)):
            final_list[i]=list1[i] if list1[i]>list2[i] else list2[i]
        
        #print(list1)
        #print(list2)
        #print(final_list)
        
        final_list2=final_list[:]
        #print("lengths",len(final_list),len(final_list2)) #check list length before deletion
        del final_list[ind2]
        del final_list[ind1]
        del final_list2[ind2]
        del final_list2[ind1]
        final_list2.insert(0,100)
        #print("lengths",len(final_list),len(final_list2))  #check list length after deletion
        return final_list,final_list2
            
    def hier_clusters(self):
        while(self.distance_mactrix.shape[0]!=self.k):
        #for i in range(5):
            s,v=np.where(self.distance_mactrix==np.min(self.distance_mactrix.min()))
            min_indices=list(zip(self.distance_mactrix.index[s],self.distance_mactrix.columns[v]))
            #print(min_indices)
            col_one_list = self.distance_mactrix[min_indices[0][0]].tolist()
            col_two_list = self.distance_mactrix[min_indices[0][1]].tolist()
            index_no1 = self.distance_mactrix.columns.get_loc(min_indices[0][0])
            index_no2 = self.distance_mactrix.columns.get_loc(min_indices[0][1])
            #print("indices:",index_no1,index_no2)
            self.distance_mactrix.drop(min_indices[0][0],inplace=True)
            self.distance_mactrix.drop(min_indices[0][1],inplace=True)
            self.distance_mactrix.drop(min_indices[0][0],axis=1,inplace=True)
            self.distance_mactrix.drop(min_indices[0][1],axis=1,inplace=True)
            #print(self.distance_mactrix.info())
            row_list_final,column_list_final=self.find_max_list(col_one_list,col_two_list,index_no1,index_no2)
            
            row_df=pd.DataFrame([row_list_final],columns=self.distance_mactrix.columns,index=[min_indices[0][0]+","+min_indices[0][1]])
            #print(row_df)
            #self.distance_mactrix.append(row_df) 
            self.distance_mactrix = pd.concat([row_df,self.distance_mactrix])
            
            #print(self.distance_mactrix.info())
            #print(self.distance_mactrix)
            self.distance_mactrix.insert(0, min_indices[0][0]+","+min_indices[0][1], column_list_final)
            #print(self.distance_mactrix.info())
            #print(self.distance_mactrix)
        #print("1 number",col_one_list)
        #print("2 number",col_two_list)
            #print(self.distance_mactrix)
        #print(self.distance_mactrix)
        for i in range(self.k):
            for loc in list(self.distance_mactrix.columns[i].split(",")):
                self.clusters[i].append((self.data.loc[int(loc)]))
        #print(self.clusters)
        
    def create_distance_matrix(self):
        for i in range(self.data.shape[0]):
            for j in range(i,self.data.shape[0]):
                if i==j:
                    self.distance_mactrix[i,j]=100
                else:
                    self.distance_mactrix[i,j]=self.cosine_distance(np.array(self.data.loc[i]),np.array(self.data.loc[j]))
                    self.distance_mactrix[j,i]=self.distance_mactrix[i,j]
        self.distance_mactrix=pd.DataFrame(self.distance_mactrix,columns=[str(i) for i in range(self.data.shape[0])],index=[str(i) for i in range(self.data.shape[0])])
        #print(self.distance_mactrix)

    def cluster_info(self):
        fp=open("agglomerative.txt","w")
        clus_dict={}
        for clus in self.distance_mactrix.columns:
            clus=[int(x) for x in clus.split(",")]
            clus.sort()
            clus_dict[clus[0]]=clus
            #final= ",".join(clus)
        for key in sorted(clus_dict.keys()):
            for id in clus_dict[key]:
                fp.write(str(id)+",")
            fp.write("\n")
        
data = pd.read_csv("COVID_3_unlabelled.csv",index_col=0)
k=0
try:
    fp=open("best_k.txt","r")
    Lines = fp.readlines()
    k=int(Lines[0])
except:
    k=5

comp_link=hierarchial(k)
comp_link.preprocess()
comp_link.create_distance_matrix()
comp_link.hier_clusters()
comp_link.plot_cluster()
comp_link.cluster_info()