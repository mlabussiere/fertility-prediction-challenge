FROM continuumio/anaconda3:2020.02

COPY environment.yml /
RUN conda env create -f /environment.yml

RUN mkdir /app
WORKDIR /app

COPY *.csv /app
COPY *.py /app
COPY *.joblib /app
COPY model.ckpt.data-00000-of-00001 /app
COPY model.ckpt.index /app
COPY model.ckpt.meta /app
COPY params_dictionary /app

ENTRYPOINT ["conda", "run", "-n", "eyra-rank", "python", "/app/run.py"]
CMD ["predict", "/data/fake_data.csv"]