FROM python:3.11

WORKDIR /app

COPY Pipfile ./
COPY Pipfile.lock ./

RUN pip install -U pip && pip install pipenv uvicorn
RUN pipenv requirements > requirements.txt
RUN pip install -r requirements.txt

COPY app/ /app/app
COPY data/ /app/data

EXPOSE 8000
