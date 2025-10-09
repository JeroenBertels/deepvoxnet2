FROM nvidia/cuda:12.9.1-base-ubuntu22.04

RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN ln -fs /usr/share/zoneinfo/Europe/Brussels /etc/localtime
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN dpkg-reconfigure -f noninteractive tzdata
RUN apt-get install -y python3.13-full
RUN python3.13 -m ensurepip --upgrade 

ARG DEEPVOXNET2_VERSION

RUN if [[ -z "$DEEPVOXNET2_VERSION" ]] ; then python3.13 -m pip install "deepvoxnet2" ; else python3.13 -m pip install "deepvoxnet2==${DEEPVOXNET2_VERSION}" ; fi

CMD ["bash"]