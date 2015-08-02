#!/usr/bin/env sh
# This scripts downloads the ptb data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

wget russellsstewart.com/s/imagenet/im_450000.h5

echo "Done."
