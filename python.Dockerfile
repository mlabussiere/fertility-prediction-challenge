FROM continuumio/anaconda3:2020.02

COPY environment.yml /
RUN conda env create -f /environment.yml

RUN mkdir /app
WORKDIR /app

COPY *.csv /app
COPY *.py /app
COPY *.joblib /app
COPY sumbissionxgb.json /app
COPY sumbissionGRU.keras /app
COPY saved_params.pkl /app


ENTRYPOINT ["conda", "run", "-n", "eyra-rank", "python", "/app/run.py"]
CMD []