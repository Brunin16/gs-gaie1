FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Gera o dataset e treina o modelo na build, para o container já iniciar pronto
RUN python src/generate_dataset.py && python src/train_pipeline.py

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
