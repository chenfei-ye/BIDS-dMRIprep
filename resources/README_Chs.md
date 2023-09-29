

# BIDS-dMRIprep

`BIDS-dmriprep` 是针对人脑磁共振diffusion MRI弥散像的预处理流程，基于[MRtrix3](https://www.mrtrix.org/)开发。主要功能包括：
- dMRI影像去噪 (*dwidenoise*)
- Gibbs环装伪影去噪 (*mrdegibbs*)
- 磁敏感伪影矫正（无需场图） (*ANTs-SyN*)
- 涡流矫正 (*dwifslpreproc*)
- mask提取 (*dwi2mask*)
- dMRI降采样（T1原始空间2mm，dMRI空间1.25mm） (*mrgrid*)
- dMRI空间转换到 T1原始空间 (*antsApplyTransforms*)
- dMRI空间转换到 MNI空间 (*antsRegistrationSyNQuick*)

数据需要符合[Brain Imaging Data Structure](http://bids.neuroimaging.io/) (BIDS)格式。

[版本历史](CHANGELOG.md)

## 本页内容
* [数据准备](#数据准备)
* [安装](#安装)
* [运行前准备](#运行前准备)
* [运行](#运行)
* [参数说明](#参数说明)
* [输出结果](#输出结果)

## 数据准备
数据需要符合[Brain Imaging Data Structure](http://bids.neuroimaging.io/) (BIDS)格式。对于`DICOM`数据文件，建议使用[dcm2bids](https://unfmontreal.github.io/Dcm2Bids)工具进行转档，参考[dcm2bids 转档中文简易使用说明](dcm2bids.md)



## 安装
本地需安装[docker](https://docs.docker.com/engine/install)，具体可参考[步骤](docker_install.md)

### 方式一：拉取镜像
```
docker pull mindsgo-sz-docker.pkg.coding.net/neuroimage_analysis/base/bids-dmriprep:latest
docker tag  mindsgo-sz-docker.pkg.coding.net/neuroimage_analysis/base/bids-dmriprep:latest  bids-dmriprep:latest
```

### 方式二：镜像创建
```
# git clone下载代码仓库
cd BIDS-dmriprep
docker build -t bids-dmriprep:latest .
```

## 运行前准备
使用[sMRIPrep](https://github.com/chenfei-ye/BIDS-sMRIprep) 完成T1影像的预处理
```
docker run -it --rm -v <bids_root>:/bids_dataset bids-smriprep:latest python /run.py /bids_dataset --participant_label 01 02 03 -MNInormalization -fsl_5ttgen -cleanup
```

## 运行
```
# 有GPU场景（调用eddy_cuda）
docker run -it --rm --gpus all -v <bids_root>:/bids_dataset bids-dmriprep:latest python /run.py /bids_dataset /bids_dataset/derivatives/dmri_prep participant -mode complete

# 无GPU场景（调用eddy_openmp）
docker run -it --rm -v <bids_root>:/bids_dataset bids-dmriprep:latest python /run.py /bids_dataset /bids_dataset/derivatives/dmri_prep participant -mode complete
```


## 参数说明
####   固定参数说明：
-   `/bids_dataset`: 容器内输入BIDS路径，通过本地路径挂载（-v）
-   `/bids_dataset/derivatives/dmri_prep`: 输出路径
-   `participant`: 个体被试水平的顺序执行

####   可选参数说明：
-   `--participant_label [str]`：指定分析某个或某几个被试。比如`--participant_label 01 03 05`。否则默认按顺序分析所有被试。
-   `--session_label [str]`：指定分析同一个被试对应的某个或某几个session。比如`--session_label 01 03 05`。否则默认按顺序分析所有session。
- `-mode ["fast", "complete"]`：是否运行涡流矫正。 complete模式会运行矫正，fast模式会跳过矫正（pipeline快捷测试场景，不推荐）。
- `-resume`：基于上一次未中断的运行结果，继续跑后续分析
- `-v`：版本查看
- `-cleanup`: 移除临时文件


## 输出结果
- 运行日志: `<local_bids_dir>/derivatives/dmri_prep/runtime.log`
- T1原始空间的预处理后dMRI影像: `<local_bids_dir>/derivatives/dmri_prep/sub-XX/dwi.nii.gz`
- MNI空间的预处理后dMRI影像: `<local_bids_dir>/derivatives/dmri_prep/sub-XX/dwi_mni.nii.gz`
- dMRI空间到T1原始空间的转换矩阵 (ANTs): `<local_bids_dir>/derivatives/dmri_prep/sub-XX/dwi_t1_ants.mat`
- dMRI空间到MNI空间的转换矩阵 (ANTs): `<local_bids_dir>/derivatives/dmri_prep/sub-XX/dwi_mni_ants.mat`
- dMRI空间到MNI空间的转换矩阵 (FSL): `<local_bids_dir>/derivatives/dmri_prep/sub-XX/dwi_mni_fsl.mat`

## Copyright
Copyright © chenfei.ye@foxmail.com
Please make sure that your usage of this code is in compliance with the code license.


