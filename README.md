# Who-art-thou
My personal dive into using deep-learning for face recognition/verification

This is something I'm building to get a better understanding of what it is like to create a facial recognition system 
that is can work in the wild (and not just a fun tutorial). The goal is to build a facial recoginition system that is
1. accurate
2. fast AF

This is build on top of timesler's [facenet-pytorch](https://github.com/timesler/facenet-pytorch) and some opencv 
to wrap it all up into a nice useable form (later we might try to get this working in the browser). 

## Setup
Install `PyTorch>=1.6` and `OpenCV` to use the repo. 

There is a hacky database that we are using to store the images for the people in the system. We load them from the `db/` 
folder and cache the results for the models to make things efficient. You can add a person by creating a new folder 
inside `db/` folder and adding images of the person. Make sure the images you add only contain the person in question.

## Usage

use `python recorder.py` to record an video with the person you want to verify. Right now it doesn't support real-time 
but will add soon. 

```
recorder.py

python recorder.py --output <output_file>

output: (default - output.avi) the name of the file saved after recording
```

Now you can run `python face_verifier.py <name>` to run the verifier which will now track and verify the person
with the name given. The name should already be in the db. 

```
face_verifier.py

python face_verifier <name> --input <input_video_file> --output <output_video_file> --recache

name: the name of the person that we need to verifiy with. This person has to be in the db. 
input: (default - output.avi) the input video file (recorded with the recoder) with person who we want to verify.
output: (default - processed.avi) this script will save the processed video showing the tracking and verification
information.
--recache - a flag to run the recaching process. The db is cached for performance so if you add new images do a recache 
before runing verifier
```

## ToDo
1. try retraining the models for verification
2. using ONNX.js, try to run this in the browser
3. try newer arch as base arch for the verification network (efficient net)
4. maybe try yolo v5 for face detection. 
