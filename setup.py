#!/usr/bin/env python
"""QuTiP-lattice: lattice support in QuTiP

QuTiP-lattice is open-source software for simulating lattice models.
QuTiP-lattice aims to provide user-friendly and efficient interface to study
lattice module using QuTiP. It was first developped as during Google Summer
of Coding 2019. QuTiP-lattice is freely available for use and/or modification
on all common platforms. Being free of any licensing fees, QuTiP-lattice is
ideal for exploring quantum mechanics in research as well as in the classroom.
"""

DOCLINES = __doc__.split('\n')

CLASSIFIERS = """\
Development Status :: 0 - Beta
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Scientific/Engineering
Operating System :: MacOS
Operating System :: POSIX
Operating System :: Unix
Operating System :: Microsoft :: Windows
"""

# import statements
import os
import sys
# The following is required to get unit tests up and running.
# If the user doesn't have, then that's OK, we'll just skip unit tests.
try:
    from setuptools import setup, Extension
    EXTRA_KWARGS = {
        'setup_require': ['pytest-runner'],
        'tests_require': ['pytest']
    }
except:
    from distutils.core import setup
    from distutils.extension import Extension
    EXTRA_KWARGS = {}

MAJOR = 0
MINOR = 1
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
REQUIRES = ['numpy (>=1.12)', 'scipy (>=1.0)', 'qutip (>=4.4)']
EXTRAS_REQUIRE = {'graphics':['matplotlib(>=1.2.1)']}
PACKAGES = ['qutip_lattice']
NAME = "qutip_lattice"
AUTHOR = ("Saumya biswas, Eric Giguere, Clemens Gneiting, Nathan Shammah")
LICENSE = "BSD"
DESCRIPTION = DOCLINES[0]
LONG_DESCRIPTION = "\n".join(DOCLINES[2:])

KEYWORDS = "quantum physics dynamics lattice"
URL = "http://qutip.org"
CLASSIFIERS = [_f for _f in CLASSIFIERS.split('\n') if _f]
PLATFORMS = ["Linux", "Mac OSX", "Unix", "Windows"]

def git_short_hash():
    try:
        git_str = "+" + os.popen('git log -1 --format="%h"').read().strip()
    except:
        git_str = ""
    else:
        if git_str == '+': #fixes setuptools PEP issues with versioning
            git_str = ''
    return git_str

FULLVERSION = VERSION
if not ISRELEASED:
    FULLVERSION += '.dev'+str(MICRO)+git_short_hash()

def write_version_py(filename='qutip_lattice/version.py'):
    cnt = """\
# THIS FILE IS GENERATED FROM QUTIP SETUP.PY
short_version = '%(version)s'
version = '%(fullversion)s'
release = %(isrelease)s
"""
    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION, 'fullversion':
                FULLVERSION, 'isrelease': str(ISRELEASED)})
    finally:
        a.close()

local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(local_path)
sys.path.insert(0, local_path)
sys.path.insert(0, os.path.join(local_path, 'qutip_lattice'))
 # to retrive _version

# always rewrite _version
if os.path.exists('qutip_lattice/version.py'):
    os.remove('qutip_lattice/version.py')

write_version_py()

setup(name = NAME,
      version = FULLVERSION,
      packages = PACKAGES,
      author = AUTHOR,
      # author_email = AUTHOR_EMAIL,
      license = LICENSE,
      description = DESCRIPTION,
      long_description = LONG_DESCRIPTION,
      keywords = KEYWORDS,
      url = URL,
      classifiers = CLASSIFIERS,
      platforms = PLATFORMS,
      requires = REQUIRES,
      extras_require = EXTRAS_REQUIRE,
      **EXTRA_KWARGS
      )
