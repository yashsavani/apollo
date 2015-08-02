#!/usr/bin/python

# import modules used here -- sys is a very standard one
import sys
import os
import os.path
import fileinput
import re
from subprocess import call, check_output

CVPATH = '/usr/local/Library/Taps/homebrew/homebrew-science/opencv.rb'

def check_install():
    brewloc = check_output(['which', 'brew']).strip()
    if not os.path.exists(brewloc):
        raise Exception('Homebrew not installed, please install Homebrew')
    
    path = os.environ['PATH']
    if 'anaconda' not in path:
        raise Exception('Anaconda is not installed, please install Anaconda')


def main():
    print "Checking: Homebrew and Anaconda installation"
    check_install()
    print ""

    print "Tapping homebrew/science"
    os.system('brew tap homebrew/science')
    print ""
    
    print "Installing: snappy leveldb gflags glog szip lmdb"
    os.system('for x in snappy leveldb gflags glog szip lmdb; do brew install -vd $x; done;')
    print ""

    print "Installing from source: protobuf boost-python"
    os.system('brew uninstall protobuf; brew install --build-from-source --with-python -vd protobuf; brew uninstall boost-python; brew install --build-from-source -vd boost boost-python')
    print ""

    print "Installing python dependencies"
    os.system('for req in $(cat requirements.txt); do pip install $req; done')
    print ""

    print "Modifying: OpenCV brew installation"
    if not os.path.exists(CVPATH):
        raise Exception('Could not find the homebrew formula for opencv.')
    dpy_lib = re.compile("(DPYTHON.*LIBRARY.*=)(.*)\"")
    dpy_inc = re.compile("(DPYTHON.*INCLUDE.*DIR.*=)(.*)\"")
    for line in fileinput.input(CVPATH, inplace=True):
        if dpy_lib.search(line) is not None :
            sys.stdout.write(dpy_lib.sub(r'\1#{py_prefix}/lib/libpython2.7.dylib" \n# \2', line))
        elif dpy_inc.search(line) is not None :
            sys.stdout.write(dpy_inc.sub(r'\1#{py_prefix}/include/python2.7" \n# \2', line))
        else:
            sys.stdout.write(line)
    print ""

    print "Installing: opencv"
    os.system('brew uninstall opencv; brew install opencv')
    print ""

    print "DONE!"

if __name__ == '__main__':
  main()
