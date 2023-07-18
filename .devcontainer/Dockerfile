FROM ubuntu:20.04
RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y gnupg2
RUN apt-get install -y wget
RUN apt-get install -y git
RUN apt-get install -y python3.9
RUN apt-get install -y curl
RUN apt-get install -y python3-distutils
RUN apt-get install -y python3-apt
RUN apt-get update
RUN apt-get install -y python3-dev
RUN apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3
RUN apt-get install -y python3-pip
RUN apt-get install -y patchelf
RUN pip3 install -U 'mujoco-py<2.2,>=2.1'
RUN pip3 install -U torch
RUN pip3 install -U pandas
RUN pip3 install -U matplotlib
RUN pip3 install -U gym==0.23.1
RUN pip3 install -U gym[mujoco]
RUN git clone https://github.com/rail-berkeley/d4rl.git
WORKDIR d4rl
RUN pip3 install -e .
WORKDIR ..
RUN curl https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz --output mujoco210.tar.gz
RUN mkdir /root/.mujoco
RUN tar -xf mujoco210.tar.gz --directory /root/.mujoco
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
ENV D4RL_SUPPRESS_IMPORT_ERROR=1

# end
CMD	sleep infinity