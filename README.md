# nishika-akutagawa

This repo is [【Nishika自社開催コンペ】AIは芥川龍之介を見分けられるのか？](https://www.nishika.com/competitions/1/summary) 2nd prize solution.

## Architecture

I use pipline lib *gokart* (https://github.com/m3dev/gokart).  

![flow](http://hoge.jpg "flow")


## Usage

1. Make data dir, set csv
```
$ mkdir data
```
https://www.nishika.com/competitions/1/data -> train.csv, test.csv

2. docker run
```
$ docker build -t nishika .
$ mkdir /tmp/resource
$ docker run -v /tmp/resource:/app/resource nishika ./run.sh
```

# Reference

- https://www.nishika.com/competitions/1/summary
