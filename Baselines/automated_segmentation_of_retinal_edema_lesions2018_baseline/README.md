### Description
This is the baseline method (MDP) for Automated Segmentation of Retinal Edema Lesions.


### Details
U-Net [1] was a popular network architecture for pixel-wise medical image segmentation in recent years. In total the network has 23 convolutional layers. The backbone consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. The contracting path follows the typical architecture of a convolutional network. It consists of the repeated application of two convolutions, each followed by a ReLU and a max pooling operation with stride 2 for down-sampling. At each down-sampling step, U-Net doubles the number of feature channels. Every step in the expansive path consists of an up-sampling of the feature map, a concatenation with the correspondingly cropped feature map from the contracting path, and two convolutions. At the final layer a 11 convolution is used to map each 64- component feature vector to the desired number of classes. The network was trained with cross entropy loss.

[1] O. Ronneberger, P. Fischer, and T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), 2015.


### Our papers about retinal edema segmentation
- Tao Wang, Zexuan Ji, Qiang Chen*, Quansen Sun*, Chenchen Yu, Wen Fan, Songtao Yuan, Qinghuai Liu. Label propagation and higher-order constraint-based segmentation of fluid-associated regions in retinal SD-OCT images. Information Sciences, 358/359: 92-111, 2016.
- Menglin Wu, Wen Fan, Qiang Chen*, Zhenlong Du, Xiaoli Li, Songtao Yuan, Hyunjin Park. Three-dimensional continuous max flow optimization-based serous retinal detachment segmentation in SD-OCT for central serous chorioretinopathy. Biomedical Optics Express, 8(9): 4257-4274, 2017
- Menglin Wu, Qiang Chen*, Xiaojun He, Ping Li, Wen Fan, Songtao Yuan, Hyunjin Park*. Automatic subretinal fluid segmentation of retinal SD-OCT images with neurosensory retinal detachment guided by enface fundus imaging. IEEE Transactions on Biomedical Engineering, 2018, 65(1):87-95.
- Zexuan Ji, Qiang Chen, Menglin Wu, Sijie Niu, Wen Fan and SongTao Yuan. Beyond Retinal Layers: A Large Blob Detection for Subretinal Fluid Segmentation in SD-OCT Images. MICCAI 2018.
