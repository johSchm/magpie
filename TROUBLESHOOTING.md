# Common Errors

##### DLib installation failed
When you encounter the error message: `dlib.cpython-37m-x86_64-linux-gnu.so: undefined symbol: cblas_dtrsm`,
the solution is as simple as specifying another `DLib` version.
It turned out that for me with my particular setup the version `19.18.0`
worked.
