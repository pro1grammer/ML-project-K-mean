def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union
    
def find_jaccard_mapping():
    for clus_id1,clus1 in k_means_cluster.items():
        max=0
        max_id=None
        for clus_id2,clus2 in hiera_clusters.items():
            val=jaccard_similarity(clus1,clus2)
            if val>max:
                max=val
                max_id=clus_id2
        cluster_mapping[clus_id1]=[max_id,max]
        hiera_clusters.pop(max_id)
k=0
try:
    fp=open("best_k.txt","r")
    Lines = fp.readlines()
    k=int(Lines[0])
except:
    k=5

fp1=open("kmeans_{}.txt".format(k),"r")
k_means_cluster={}
i=1
for line in fp1:
    line = line.strip()
    l_arr=list(line.split(","))
    k_means_cluster["k_means_cluster_{}".format(i)]=l_arr
    i+=1

fp2=open("agglomerative.txt","r")
hiera_clusters={}
i=1
for line in fp2:
    line = line.strip()
    l_arr=list(line.split(","))
    hiera_clusters["heirarchial_cluster_{}".format(i)]=l_arr
    i+=1

cluster_mapping={}
find_jaccard_mapping()

print("Jaccard Similarity Scores:-")
print()
print("K-means Cluster ID\tHeirarchial Cluster ID\tJaccard Similarity Score")
print("-------------------------------------------------------------------------")
for key,val in cluster_mapping.items():
    print("{}\t{}\t{}".format(key,val[0],val[1]))