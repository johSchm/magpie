<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Common Errors](#common-errors)
        - [DLib installation failed](#dlib-installation-failed)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Common Errors

##### DLib installation failed
When you encounter the error message: `dlib.cpython-37m-x86_64-linux-gnu.so: undefined symbol: cblas_dtrsm`,
the solution is as simple as specifying another `DLib` version.
It turned out that for me with my particular setup the version `19.18.0`
worked.
