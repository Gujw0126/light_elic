# light_elic
## 1. 模型编解码各步骤的时间占比  
一个超先验压缩模型通常由y_a,y_s,h_a,h_s,熵编码，熵解码四个主要组成部分。 
![image](https://github.com/Gujw0126/light_elic/blob/main/resource2/hyper.png) 
为了提高现有AI压缩模型的吞吐率，现对这六个部分的时间占比进行分析，结果如下图所示。可以观察到rANS编解码时间占据了80%以上的运行时间。   
![image](https://github.com/Gujw0126/light_elic/blob/main/resource2/time_pro.png)   

## 2. 通道冗余现象
模型中每个通道的重要程度并不相同，因此在编码层面，一些通道存在冗余。即对这些通道是否编码不会对模型的压缩性能产生较大的影响。以ELIC模型为例，不同通道的熵和对PSNR的贡献程度不尽相同。但是可以观察到通道熵越大，对PSNR的贡献越大。  
![image](https://github.com/Gujw0126/light_elic/blob/main/resource2/channel_entropy_original.png)    


为获得每个通道的重要程度，在模型中引入缩放单元和反缩放单元，y经过缩放后再经过熵编码模块。一般来说，通道对应的缩放单元系数越大，该通道越重要。将通道的熵和PSNR图按照缩放系数升序排序可得到较好的递增趋势。  
![image](https://github.com/Gujw0126/light_elic/blob/main/resource2/hyper_gain_entropy.png)   
![image](https://github.com/Gujw0126/light_elic/blob/main/resource2/PSNR_mask.png)  

将每个通道实际编码数值可视化，得到以下结果。可见缩放系数小的通道，编码值较小，编码数量也较少
![image](https://github.com/Gujw0126/light_elic/blob/main/resource2/encode_results.png)

## 3. hyperprior模型通道跳跃加速

缩放系数小的通道可用超先验预测的均值代替其编码值，从而跳过熵编码与解码阶段，从而降低编解码的延迟。  
经实验，在Hyperprior模型上，最低档位的模型在基本不影响压缩性能的情况下可跳跃60%以上的通道，将编解码延迟降低25%  
![image](https://github.com/Gujw0126/light_elic/blob/main/resource2/times.png)  


