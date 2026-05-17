@echo off
clang example.c src/util.c -gfull -fsanitize=address -o m.exe 
@echo on
