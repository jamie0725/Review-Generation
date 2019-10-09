# docker build -t thashim/condeditor .
FROM kelvinguu/textmorph:1.2
MAINTAINER Tatsu Hashimoto <thashim@mit.edu>

USER root
WORKDIR /
RUN apt-get -y update

#needed python packages
RUN pip install regex editdistance
