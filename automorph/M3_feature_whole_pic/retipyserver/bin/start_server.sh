#!/bin/sh

export FLASK_APP=retipyserver
export FLASK_DEBUG=true
exec flask run
