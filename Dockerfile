FROM tensorflow/tensorflow:2.7.0-gpu

RUN python3 -m pip install --upgrade pip

ARG DEEPVOXNET2_VERSION

RUN if [[ -z "$DEEPVOXNET2_VERSION" ]] ; then python3 -m pip install "deepvoxnet2[sitk]" ; else python3 -m pip install "deepvoxnet2[sitk]==${DEEPVOXNET2_VERSION}" ; fi

CMD ["bash"]