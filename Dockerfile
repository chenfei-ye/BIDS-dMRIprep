FROM flywheel/fsl-base:6.0.6-focal
MAINTAINER Chenfei <chenfei.ye@foxmail.com>

RUN apt update && apt-get install -y \
	build-essential \
	tar \
    wget \
	curl \
	git \
	unzip \
	openssh-server
	
# install python packages 
RUN pip install -i https://mirrors.aliyun.com/pypi/simple/ dipy \
	nilearn \
	pandas \
	matplotlib \
	nipype \
	pybids \
	pillow \
	scikit-image \
	numpy \
	nibabel
	
# download precompiled ANTs
RUN curl -SL -o /opt/ants_compiled_ubuntu16.tar.gz http://api.open.brainlabel.org/data/ycf/ants_compiled_ubuntu16.tar.gz && \
	tar -xzvf /opt/ants_compiled_ubuntu16.tar.gz -C /opt && \
	rm /opt/ants_compiled_ubuntu16.tar.gz

# Install c3d
RUN wget -O /c3d-nightly-Linux-gcc64.tar.gz http://api.open.brainlabel.org/data/ycf/c3d-nightly-Linux-gcc64.tar.gz && \
	tar -xzvf /c3d-nightly-Linux-gcc64.tar.gz -C /opt && \
    rm /c3d-nightly-Linux-gcc64.tar.gz 
	
# Download MRtrix3 (3.0.2)
RUN wget -O /mrtrix3_v3.0.2.zip http://api.open.brainlabel.org/data/ycf/mrtrix3_v3.0.2.zip && \
    unzip -d /opt/mrtrix3 /mrtrix3_v3.0.2.zip && \
	rm /mrtrix3_v3.0.2.zip

ENV PATH="/opt/ants-latest/bin:/opt/mrtrix3/bin:/opt/c3d-1.3.0-Linux-gcc64/bin:$PATH" \
	ANTSPATH="/opt/ants-latest/bin" \
    LD_LIBRARY_PATH="/opt/ants-latest/lib:$LD_LIBRARY_PATH" \
    OS=Linux 
	
ADD ./ /
ADD ./data /data/
RUN chmod -R 777  /opt/mrtrix3  && \
	echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
	echo "root:mindsgo123" | chpasswd && \
	apt-get clean && \
	rm -rf /external /root/.cache/pip && \
	rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* 
	
CMD ["python3", "/run.py"]


