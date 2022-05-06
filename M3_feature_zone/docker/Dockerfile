FROM archlinux/base:latest

LABEL maintainer="Alejandro Valdes <alejandrovaldes@live.com>"

RUN echo 'Server = http://mirrors.udenar.edu.co/archlinux/$repo/os/$arch' > /etc/pacman.d/mirrorlist

RUN pacman -Sy \
    intel-tbb \
    openexr \
    gst-plugins-base \
    libdc1394 \
    cblas \
    lapack \
    libgphoto2 \
    jasper \
    ffmpeg \
    cmake \
    python-numpy \
    python2-numpy \
    mesa \
    eigen \
    hdf5 \
    lapacke \
    gtk3 \
    vtk \
    glew \
    python-pip \
    tensorflow \
    python-tensorflow \
    --noconfirm

RUN pacman -Sy opencv --noconfirm

ENV NB_USER retipy
ENV NB_UID 1000
ENV LC_ALL en_US.utf-8
ENV LANG en_US.utf-8
ENV RETIPY_HOME /home/retipy/src

RUN groupadd --system --gid 1000 retipy && \
    useradd -m -s /bin/zsh -N -u ${NB_UID} --gid retipy ${NB_USER} && \
    mkdir -p /src && \
    chown ${NB_USER} /src -R  && \
    mkdir -p /opt/retipy && \
    chown ${NB_USER} /opt/retipy -R

# RUN localedef -i en_US -f UTF-8 en_US.UTF-8

USER retipy

RUN pip install --user flask gunicorn matplotlib pillow scikit-image scikit-learn scipy numpy h5py

ADD docker/matplotlibrc /home/retipy/.config/matplotlib/

COPY --chown=retipy:retipy . /opt/retipy

# install retipy library
RUN pip install --user -e /opt/retipy/retipy/ && \
    mkdir -p ${RETIPY_HOME} && \
    cp /opt/retipy/retipy/*.py ${RETIPY_HOME} && \
    cp -r /opt/retipy/retipy/resources ${RETIPY_HOME} && \
    sed -i 's/\.\.\///' ${RETIPY_HOME}/resources/retipy.config && \
    mkdir ${RETIPY_HOME}/build && \
    rm ${RETIPY_HOME}/setup.py

# install retipy server
RUN pip install --user -e /opt/retipy/retipyserver/

ENV PATH /home/retipy/.local/bin:${PATH}
ENV FLASK_APP retipyserver

EXPOSE 5000

CMD ["gunicorn", "--log-level", "debug", "-b", "0.0.0.0:5000", "-w", "2", "-t", "300", "retipyserver:app"]
