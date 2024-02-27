## Quick Start
If you are Ashley and have forgotten how to run this code on cedar then follow these instructions 

```
source mae_env/bin/activate
```

```
cd scratch/github/TileSlicer/
```

```
module load rust
```

```
module load gcc/9.3.0 arrow/13.0.0 python/3.10
```

```
pip install -r requirements.txt 
```

make sure 

```
platform = 'cedar'
```

and then run
```
python data_stream_test.py
```
to test it all works