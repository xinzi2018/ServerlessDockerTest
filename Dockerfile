FROM python:3.6

WORKDIR ./ServerLessDockerTest
 
ADD . .  

RUN pip install -r requirements.txt

CMD ["python", "./helloworld.py"]