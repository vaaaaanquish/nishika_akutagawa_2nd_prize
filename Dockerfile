FROM python:3.6.8-stretch

WORKDIR /app

RUN apt update &&\
    rm -rf ~/.cache &&\
    apt clean all

# mecab
RUN apt install -y mecab libmecab-dev mecab-ipadic mecab-ipadic-utf8 gsutil libwww-perl
RUN curl -sSL https://sdk.cloud.google.com | bash
ENV PATH /root/google-cloud-sdk/bin:$PATH
RUN mkdir -p /usr/lib/x86_64-linux-gnu/mecab
RUN ln -s /var/lib/mecab/dic /usr/lib/x86_64-linux-gnu/mecab/dic

# juman
RUN wget https://github.com/ku-nlp/jumanpp/releases/download/v2.0.0-rc3/jumanpp-2.0.0-rc3.tar.xz
RUN tar xvf jumanpp-2.0.0-rc3.tar.xz
RUN apt install build-essential -y
RUN apt install cmake -y
WORKDIR /app/jumanpp-2.0.0-rc3/
RUN mkdir build
WORKDIR /app/jumanpp-2.0.0-rc3/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
RUN make
RUN make install

# python
WORKDIR /app
RUN pip install --upgrade pip &&\
    rm -rf ~/.cache
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
ENTRYPOINT python

# files
WORKDIR /
COPY ./conf /app/conf
COPY ./data /app/data
COPY ./nishika /app/nishika
COPY ./main.py /app/main.py
COPY ./run.sh /app/run.sh

WORKDIR /app
ENTRYPOINT [ "/bin/bash" ]
VOLUME "/app"
