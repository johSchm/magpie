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


import cmd.arg_parse as arg_parse
import cmd.ansi as ansi


print("\n")
print(f">> {ansi.BColors.BOLD}welcome to magpie{ansi.BColors.ENDC} <<")
print("Created by J. Schmidt.")
print("https://github.com/johSchm/magpie")
print("\n")
args = arg_parse.parse_args()


