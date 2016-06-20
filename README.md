# FaceDetect
The program recursively traverses the specified directory on the disk looking for files with photos, detects on their faces, 
eyes and mouth. Person found transformerait and saves to disk in format .jpg, also save the file with the coordinates of found
faces in the format .json. The program implemented a pool of threads each thread which handles your photo.


The program was developed on C++ in an environment in VS2015 using the library OpenCV(2.4.13) and Boost(1_60_0).
