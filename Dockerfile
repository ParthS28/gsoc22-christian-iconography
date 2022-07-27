FROM ubuntu:20.04

RUN apt-get update

RUN apt-get install --assume-yes --no-install-recommends --quiet \
        python3 \
        python3-pip \
        ffmpeg

RUN pip install --no-cache --upgrade pip setuptools

WORKDIR /EmileMale

ADD ./EmileMaleV1/ .

RUN ls -a 

RUN pip3 install -r requirements.txt


ENTRYPOINT ["./run.sh"]