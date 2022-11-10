FROM tensorflow/tensorflow:2.7.0-gpu

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install 'deepvoxnet2[sitk]'

CMD ["bash"]