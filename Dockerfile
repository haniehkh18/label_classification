FROM python:latest
RUN mkdir /home/Hanieh/label_classification
COPY . /home/hanieh/Hanieh/label_classification
WORKDIR /home/hanieh/Hanieh/nlp_assignment
#RUN pip install -r src/requirements.txt
WORKDIR /home/hanieh/Hanieh/nlp_assignment/src
CMD ["python", "flask_api.py"]