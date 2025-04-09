# RPX
The repository for Robot Perception X dataset and benchmarking

## Motion-Vectors
```shell
# stich png to make mp4
ffmpeg \
-framerate 15 \
-pattern_type glob -i rgb/"*.png" \
-c:v libx264 \
-preset slow \
-tune animation \
-crf 18 \
output264.mp4 

# install package
pip install motion-vector-extractor 

# extract mvs
extract_mvs output264.mp4 --preview --verbose --dump
```