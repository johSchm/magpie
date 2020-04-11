#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" -------------------------------------------
@author:     Johann Schmidt
@date:       2020
@refs:
@todo:
@bug:
@brief:      The magpie main.
------------------------------------------- """


import src.cmd.arg_parse as arg_parse
import src.cmd.ansi as ansi
import colorama

colorama.init()

print("\n")
print("welcome to magpie")
print("Created by J. Schmidt.")
print("https://github.com/johSchm/magpie")
print("\n")
args = arg_parse.parse_args()


