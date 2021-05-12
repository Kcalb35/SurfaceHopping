# Surface Hopping

This project including FSSH, PC-FSSH, PC-BCSH

## how to build

```bash
git clone git@github.com:Kcalb35/SurfaceHopping.git
cd SurfaceHopping/lib
git clone git@github.com/ToruNiina/toml11
git clone git@github.com:amrayn/easyloggingpp.git
cd ..
mkdir build
cd build
cmake .. && make
```

## how to run

1. create `log.conf` according to [easyloggin++ docs](https://github.com/amrayn/easyloggingpp#using-configuration-file)
    ```
   *GLOBAL:
      FORMAT ="%datetime %msg"
      FILENAME="data/fssh.log"
      ENABLED=true
      TO_FILE=true
      TO_STANDARD_OUTPUT = true
   ```

2. create `sh.toml` for configurations
   ```
   [shared]
   model = 2
   state = 0
   cores = 4
   dt = 0.1
   count = 1000
   method = "PCBCSH"
   norm = true
   debug = false
   save_path = "serial.dat"

   [single]
   momenta = 7.0

   [serial]
   start = 31.0 
   end = 50.0 
   interval = 1.0

   [multi]
   momenta = [5.0, 7.0, 8.5, 9.5, 10.5, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]	
```
   
   | configuration name | description                                          |
   | ------------------ | ---------------------------------------------------- |
   | model              | 1-SAC 2-DAC 3-ECR                                    |
   | state              | 0-lower PES 1-high PES                               |
   | cores              | multicores proccessing                               |
   | dt                 | simulate interval                                    |
   | count              | simulation turns for each momenta                    |
   | method             | FSSH PCFSSH PCBCSH                                   |
   | norm               | whether enable normalization distribution in momenta |
   | debug              | whether print debug info                             |
   | save_path          | result saving path                                   |
   
3. using `./SH sh.toml` to run