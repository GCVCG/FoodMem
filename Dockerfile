FROM max810/xmem2:gui

RUN apt-get -y update && apt-get install -y wget imagemagick parallel

RUN pip3 install -U openmim
RUN mim install mmcv-full==1.7.1
RUN pip3 install -r requirements.txt