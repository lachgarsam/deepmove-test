FROM python:3
RUN mkdir -p /app
WORKDIR /app
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
COPY ./2.png ./1.png ./image_test.png ./model_DeepMove.h5  ./requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY ./app.py /app/
COPY ./static/ /app/static/
COPY ./templates/ /app/templates/
CMD python app.py
#ENTRYPOINT ["python", "app.py"]