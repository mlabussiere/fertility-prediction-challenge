FROM continuumio/anaconda3:4.1.1

COPY environment.yml /
RUN conda env create -f /environment.yml

RUN mkdir /app
WORKDIR /app

COPY *.csv /app
COPY *.py /app
COPY *.joblib /app

ENTRYPOINT ["conda", "run", "-n", "env_submission", "python", "/app/run.py"]
CMD ["predict", "/data/fake_data.csv"]