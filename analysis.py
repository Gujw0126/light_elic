import numpy as np
import torch
from torch import Tensor
from sklearn.decomposition import PCA
import sklearn.cluster as sk_cluster
import sklearn.cluster as cluster
import os
from teacher_models import ELICHyper, ELICExtract
from compressai.compressai.zoo import load_state_dict
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn.metrics import pairwise_distances

 
def get_PCA(X:Tensor, save:bool=False, savepath:str=None):
    """
    X:Tensor, latent_y [1,C,H,W]
    """
    B,C,H,W = X.shape
    data = X[0]
    data = data.reshape(C,H*W).detach().to("cpu").numpy()
    
    data_mean = np.mean(data, axis=0)
    data_scale = np.std(data, axis=0)
    data_normalized = (data-data_mean)/(data_scale+1e-7)
    pca = PCA(n_components=3)
    X_reduced = pca.fit_transform(data_normalized)
    if save:
        file_path = os.path.join(savepath, "pca.npy")
        np.save(file=file_path, arr=X_reduced)


def plot_pca_result(pca_result:str):
    np_result = np.load(pca_result)
    x = np_result[:,0]
    y = np_result[:,1]
    z = np_result[:,2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.savefig("./pca_results/plt_result.png")
    print("loading")


def cosine_similarity(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

cosine_metric = lambda x, y : 1 - cosine_similarity(x,y)



class KMeansAnalysis:
    def __init__(self, k):
        self.kmeans = sk_cluster.KMeans(n_clusters=k, random_state=6, n_init=10)
        self.y_hat = None
        self.centers = None
        self.k = k
    
    def kmeans_cluster(self, X:Tensor):
        """
        X:model的compress函数跑完后的latent, shape[B,C,H,W]
        聚类结果存在self.y_hat里面
        return None
        """
        B,C,H,W = X.shape
        data = X[0]
        data = data.reshape(C,H*W).detach().to("cpu").numpy()
        #normalize
        data_division = np.zeros_like(data)
        for ch_idx in range(C):
            data_division[ch_idx,:] = np.linalg.norm(data[ch_idx])

        #data_mean = np.mean(data, axis=0)
        #data_scale = np.std(data, axis=0)
        #data_normalized = (data-data_mean)/(data_scale+1e-7)
        data_normalized = data/(data_division+1e-7)
        print(np.linalg.norm(data_normalized[34]))
        self.kmeans.fit(data_normalized)
        self.y_hat = self.kmeans.predict(data_normalized)
        self.centers = self.kmeans.cluster_centers_
        print("predict over")


    def distribute_cluster(self, X, out_dir):
        """
        X:tensor ,out_compress["y"]
        out_dir:empty dirctory
        用于将X的聚类结果分发到out_dir的各类子文件下, 聚类中心也保存到各类子文件下
        """
        B,C,H,W = X.shape
        for i in range(self.k):
            os.mkdir(os.path.join(out_dir,"class_{}".format(i)))
            center_map = self.centers[i]
            center_map = center_map/np.max(center_map)
            #plt.figure(figsize=(8,4))
            #plt.imshow(center_map)
            savename = os.path.join(os.path.join(out_dir, "class_{}".format(i)),"center_{}.png".format(i))
            #plt.savefig(savename)
            #plt.close()            
    
        for i in range(C):
            latent_map = X[0,i,:,:].cpu().detach().numpy()
            latent_map_norm = latent_map/np.max(latent_map)
            plt.figure(figsize=(8,4))
            plt.imshow(latent_map_norm)
            savename = os.path.join(os.path.join(out_dir, "class_{}".format(self.y_hat[i])),"channel_{}.png".format(i))
            plt.savefig(savename)
            plt.close()
    

    def plot_index(self, out_dir):
        cluster_group = []
        for i in range(self.k):
            cluster_group.append([])
        channel_num = self.y_hat.size
        for ch_idx in range(channel_num):
            cluster_group[self.y_hat[ch_idx]].append(ch_idx)
        
        color_map = np.zeros(self.k)
        for group in range(self.k):
            color_map[group] = min(cluster_group[group]) if len(cluster_group[group])>0 else 0

        cluster_map = np.zeros((channel_num, 20))

        for g in range(self.k):
                group = cluster_group[g]
                for ch_idx in group:
                    cluster_map[ch_idx][:]=color_map[g]

        plt.figure(figsize=(8,10))
        plt.imshow(cluster_map, cmap='hot')
        plt.savefig(out_dir)
        plt.close()




class DBSCANAnalysis:
    def __init__(self, eps, min_sample, my_metric):
        self.dbscan = cluster.DBSCAN(eps=eps, min_samples=min_sample, metric=my_metric)
        self.y_hat = None
    
    def dbscan_cluster(self, X:Tensor):
        """
        X:model的compress函数跑完后的latent, shape[B,C,H,W]
        聚类结果存在self.y_hat里面
        return None
        """
        B,C,H,W = X.shape
        data = X[:,:,0,0].permute(1,0).detach().to("cpu").numpy()
        #data = data.reshape(C,H*W).detach().to("cpu").numpy()
        data_mean = np.mean(data, axis=0)
        data_scale = np.std(data, axis=0)
        data_normalized = (data-data_mean)/(data_scale+1e-7)
        self.y_hat = self.dbscan.fit_predict(data_normalized)
        print("predict over")
    
    def plot_index(self, out_dir):
        channel_num = self.y_hat.size
        x = np.arange(0,channel_num, 1)
        y = np.zeros_like(x)
        for ch_idx in range(channel_num):
            y[ch_idx] = self.y_hat[ch_idx]
        plt.scatter(x=x,y=y,s=1)
        plt.savefig(out_dir)
        plt.close()

    


if __name__=="__main__":
    torch.manual_seed(6)
    random.seed(6)
    torch.cuda.manual_seed_all(6)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    checkpoint = "/mnt/data3/jingwengu/ELIC_light/hyper/lambda2/correct.pth.tar"
    device = "cuda:1"
    state_dict = load_state_dict(torch.load(checkpoint,map_location=device))
    #extract_weight = state_dict["g_a.15.weight"]
    print("find extract weight")
    
    
    model_cls = ELICHyper()
    model = model_cls.from_state_dict(state_dict).to(device).eval()
    img_path = "/mnt/data1/jingwengu/kodak/kodim02.png"
    img = Image.open(img_path).convert("RGB")
    trans = transforms.ToTensor()
    img_tensor = trans(img).unsqueeze(0).to(device)
    print("start compress")
    out_compress = model.compress(img_tensor)
    #TODO:在C方向做归一化
    cluster_160 = KMeansAnalysis(k=20)
    cluster_160.kmeans_cluster(X=out_compress["latent"].cpu())
    #cluster_160.distribute_cluster(X=out_compress["latent"].cpu(), out_dir='/mnt/data3/jingwengu/ELIC_light/hyper_cluster/kmeans/lambda2/k_160_kodim01')
    cluster_160.plot_index(out_dir="./kmeans_l2/center_20/kodim02_20_cluster.png")
    """
    print("start fitting")
    get_PCA(X=out_compress["latent"].cpu(), save=True, savepath="./pca_results")
    plot_pca_result("./pca_results/pca.npy")
   
    data = extract_weight[:,:,0,0].permute(1,0).cpu()
    print(cosine_similarity(data[0],data[2]))
    cluster_160 = DBSCANAnalysis(eps=0.8, min_sample=1,my_metric=cosine_metric)
    cluster_160.dbscan_cluster(X=extract_weight.cpu())
    cluster_160.plot_index(out_dir="./dbscan_cos/index4_weight.png")
    """