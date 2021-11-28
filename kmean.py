from math import e as e
import pandas as pd
import numpy as np
from math import log
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random as rd


class k_means:
    def __init__(self,k=3,iter=20):
        self.k=k
        self.data=data
        self.iter=iter
        self.centroids=[]
        self.clusters={}
        self.closest_cluster={}
        
    def preprocess(self):
        shape=self.data.shape
        for column in self.data:
            mean=self.data.mean()[column]
            std=self.data.std()[column]
            for i in range(shape[0]):
                self.data.at[i,column]=(self.data.at[i,column]-mean)/std
        
        
    def init_centroids(self):
        rows=self.data.shape[0]
        centroid_indices = rd.sample(range(0, rows), self.k)
        #print(centroid_indices)
        centroids = []
        for i in centroid_indices:
            self.centroids.append(self.data.loc[i])
        self.centroids=np.array(self.centroids)
        #print("Initial Centroids:\n",self.centroids)
   
    def cosine_distance(self,a,b):
        assert len(a) == len(b)
        ab_sum, a_sum, b_sum = 0, 0, 0
        for ai, bi in zip(a, b):
            ab_sum += ai * bi
            a_sum += ai * ai
            b_sum += bi * bi
        return 1-ab_sum / sqrt(a_sum * b_sum)
        
    def eucl_distance(self,a,b):
        sum=0
        for ai,bi in zip(a,b):
            diff=ai-bi
            sqr=diff**2
            sum+=sqr
        return sqrt(sum)
        
    def min_dis(self,dis_list):
        min=101 #as max value is 100
        i=-1
        index=-1
        for dis in dis_list:
            i+=1
            if min>dis:
                min=dis
                index=i
        return index,min
        
    def update_centroid(self,clus):
        for cluster_id,dis_list in clus.items():
            mean=pd.DataFrame(dis_list).mean()  #may use median here
            self.centroids[cluster_id]=mean
            self.centroids=np.array(self.centroids)
            
                
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
        plt.title("K-Means Clustering")
        
        plt.show()
  
    def clustering(self):
        for iter in range(self.iter):
            clus={new_list: [] for new_list in range(self.k)}
            for i in range(self.data.shape[0]):
                dis_list=[]
                for k in range(self.k):
                    dis_list.append(self.cosine_distance(np.array(self.data.loc[i]),self.centroids[k])) #may use euclidean distance
                cluster_id,min_distance=self.min_dis(dis_list)
                clus[cluster_id].append((self.data.loc[i]))
            self.update_centroid(clus)
        self.clusters=clus
        
    def find_closest_cluster(self):
        clus={new_list: [] for new_list in range(self.k)}
        i=0
        for clus_center in self.centroids:
            dis_list=[]
            cent=[]
            for clus_center_2 in self.centroids:
                if (clus_center!=clus_center_2).all():
                    dis_list.append(self.cosine_distance(clus_center,clus_center_2))
                    cent.append(clus_center_2)
            min_index,min_distance=self.min_dis(dis_list)
            clus[i]=cent[min_index]
            i+=1
        self.closest_cluster=clus
        #print(self.closest_cluster)
        '''for a,b in zip(self.centroids,self.closest_cluster.values()):
            print(a,"-->",b)'''
      
    def mean_distance(self,one_row,clus):
        dis=0
        for i in clus.index.tolist():
            dis+=(self.cosine_distance(np.array(one_row),np.array(clus.loc[i])))
        return dis/clus.shape[0]
      
    def index_cluster(self,clus):
        ind=0
        for i in self.centroids:
            if (i==np.array(clus)).all():
                return ind
            ind+=1
      
    def silhoute_coeff(self):
        self.find_closest_cluster()
        mean_s=0
        j=0
        for cluster in self.clusters.values():
            cluster=pd.DataFrame(cluster)
            s=0
            closest=self.index_cluster(self.closest_cluster[j])
            j+=1
            for i in cluster.index.tolist():
                a=self.mean_distance(cluster.loc[i],cluster)
                b=self.mean_distance(cluster.loc[i],pd.DataFrame(self.clusters[closest]))
                #print(a,b)
                max=0
                if a>b:
                    max=a
                else:
                    max=b
                s=(b-a)/max
                #print(s)
                mean_s+=s
        return mean_s/self.data.shape[0]
            
        
    def cluster_info(self,file):
        fp=open("kmeans_{}.txt".format(file),"w")
        clus_dict={}
        for cluster in self.clusters.values():
            cluster=pd.DataFrame(cluster)
            clus=sorted(cluster.index.tolist())
            clus.sort()
            clus_dict[clus[0]]=clus
        for key in sorted(clus_dict.keys()):
            for id in clus_dict[key]:
                fp.write(str(id)+",")
            fp.write("\n")
        
data = pd.read_csv("COVID_3_unlabelled.csv",index_col=0)
#print(data.info())
#print(data.mean())
#print(data)
#print(data.describe())

silhoute_coeff_dict={}

cluster=k_means()
cluster.preprocess()
cluster.init_centroids()
cluster.clustering()
cluster.plot_cluster()
cluster.cluster_info(3)
silhoute_coeff_dict[3]=cluster.silhoute_coeff()
print("At k= 3 , Silhouette Coefficient: ",silhoute_coeff_dict[3])

for i in range(4,7):
    cluster=k_means(i)
    cluster.preprocess()
    cluster.init_centroids()
    cluster.clustering()
    cluster.plot_cluster()
    cluster.cluster_info(i)
    silhoute_coeff_dict[i]=cluster.silhoute_coeff()
    print("At k=",i,", Silhouette Coefficient: ",silhoute_coeff_dict[i])
    
best_k=0
max_silh_coeff=-1
for key,value in silhoute_coeff_dict.items():
    if value>max_silh_coeff:
        max_silh_coeff=value
        best_k=key
        
print()
print("The best clustering is reached at k= {} at a value of Silhouette Coefficient: {}".format(best_k,max_silh_coeff))
fp=open("best_k.txt","w")
fp.write(str(best_k))