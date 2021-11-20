#Base Image to use
FROM python:3.9

#Expose port 8080
EXPOSE 8080

#Optional - install git to fetch packages directly from github
RUN apt-get update && apt-get install -y git
RUN apt-get update && apt-get install -y pip

RUN apt-get install -y portaudio19-dev python-all-dev python3-all-dev
RUN pip install pyaudio
#RUN apt-get install portaudio19-dev
#RUN apt-get install portaudio

#RUN apt-get install python-pyaudio

#RUN apt-get update && apt-get install -y brew install portaudio

#Copy Requirements.txt file into app directory
COPY requirements.txt app/requirements.txt

#install all requirements in requirements.txt
RUN pip install -r app/requirements.txt

#Copy all files in current directory into app directory
COPY . /app

#Change Working Directory to app directory
WORKDIR /app

#Run the application on port 8080
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]