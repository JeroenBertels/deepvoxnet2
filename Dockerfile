FROM tensorflow/tensorflow:2.7.0-gpu
RUN python -m pip --upgrade pip
RUN python -m pip install 'deepvoxnet2[sitk]'
CMD ["bash"]