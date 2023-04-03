## example command used for `lib/` generation 
## after generation of the library the following line `add_subdirectory(src/test)` was removed from the `lib/CMakeLists.txt` file
```
python3.9 main.py --targets fpga_native fpga_generic --no-workaround-warnings --no-concepts -o ../tester/lib
```
