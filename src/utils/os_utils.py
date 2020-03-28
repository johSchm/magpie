#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
author:     Johann Schmidt
date:       October 2019
------------------------------------------- """


from sys import platform
from enum import Enum
import os
import sys


class OperatingSystems(Enum):
    """ Enum of supported OSs.
    """
    LINUX = "linux"
    WIN = "windows"
    OSX = "osx"


class PythonVersion(Enum):
    """ Enum of Python versions.
    """
    PYTHON_2 = 2
    PYTHON_3 = 3


def get_operation_system():
    """ Returns the current OS.
    :return: OS @see OperatingSystems
    """
    if platform == "linux" or platform == "linux2":
        return OperatingSystems.LINUX
    elif platform == "darwin":
        return OperatingSystems.OSX
    elif platform == "win32":
        return OperatingSystems.WIN
    else:
        raise NotImplementedError("Other operating systems are not yet supported!")


def python_version():
    """ Returns the current python version.
    :return: python version
    """
    if sys.version_info > (3, 0):
        return PythonVersion.PYTHON_3
    else:
        return PythonVersion.PYTHON_2
