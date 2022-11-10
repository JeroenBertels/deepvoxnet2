FROM tensorflow/tensorflow:2.7.0-gpu
RUN pip install 'deepvoxnet2[sitk]'
CMD ["bash"]