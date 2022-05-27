retipy-python
======
[![Build Status](https://travis-ci.org/alevalv/retipy-python.svg?branch=master)](https://travis-ci.org/alevalv/retipy-python)
[![Coverage Status](https://codecov.io/gh/alevalv/retipy-python/branch/master/graph/badge.svg)](https://codecov.io/gh/alevalv/retipy-python)

retipy-python is part of the [retipy](https://github.com/alevalv/retipy) project.

The goal of this project is to create a python library to perform different image processing operations on fundus retinal images. Currently there are vessel segmentation, bifurcation detection and tortuosity measures available as a REST endpoints.

Installation
------------

### Development Environment

This project uses [OpenCV](https://opencv.org/) 4.0.0. Any version older than 4 will not work.

To use this project locally and be able to make changes to the retipy code, you can run the following command in
your console (having python3 and pip installed):

```bash
pip install --user -e .
```

This command should be ran inside the src folder that contains the retipy folder. It will make the retipy
library available to the user that ran it.

### Docker

The library is also available as a docker container at [alevalv/retipy-python](https://hub.docker.com/r/alevalv/retipy-python/):

```bash
docker pull alevalv/retipy-python:latest
```
By default, the docker image will expose a REST endpoint in port 5000.

License
-------
retipy is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
