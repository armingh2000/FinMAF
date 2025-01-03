FROM continuumio/miniconda3

RUN conda config --prepend channels pkgs/main
RUN conda config --prepend channels conda-forge
RUN conda update --all

RUN mkdir /FinMAF/
COPY environment.yml /FinMAF/environment.yml
COPY models /FinMAF/models
COPY src /FinMAF/src

RUN conda env create --file /FinMAF/environment.yml
RUN echo "conda activate FinMAF" >> ~/.bashrc
ENTRYPOINT ["bash"]

WORKDIR /FinMAF