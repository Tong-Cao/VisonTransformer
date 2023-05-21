FROM python:3.8

RUN mkdir /ct_ViT

COPY requirements.txt /ct_ViT

WORKDIR /ct_streamlit

RUN pip install -r requirements.txt

COPY . /ct_ViT

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_show.py"]