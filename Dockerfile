FROM myriad_cv4
# Install system packages
RUN wget -q -O /tmp/get-pip.py --no-check-certificate https://bootstrap.pypa.io/get-pip.py && python3 /tmp/get-pip.py
RUN  pip install -U pip \
 numpy \
 pillow \
 matplotlib \
 notebook \
 Flask \
 imutils \
 paho-mqtt \
 PyYAML

# Minimize image size 
RUN (apt-get autoremove -y; \
     apt-get autoclean -y)


WORKDIR /opt/frigate/
ADD frigate frigate/
ADD config config/
ADD frigate/model frigate/model/
COPY detect_objects.py .
COPY setupvars.sh /usr/local/bin/
RUN ln -s usr/local/bin/setupvars.sh
ENTRYPOINT ["setupvars.sh"]

