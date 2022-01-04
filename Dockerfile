FROM python:3.8

WORKDIR /app

COPY . .

RUN pip install -r ./requirements.txt
RUN pip install scikit-image

EXPOSE 8501

CMD ["streamlit","run","./app.py"]