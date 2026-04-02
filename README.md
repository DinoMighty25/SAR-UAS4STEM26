# SAR-UAS4STEM26
Some Assembly Required's GitHub repository for the 25-26 UAS4STEM season

## Current progress (4/1/26)
Files have been split into a precision land, QR decode, and main file under /src. The /models directory contains the .rpk model used in the current decode QR code script. Another model, this time trained for instance segmentation, was uploaded and converted to the IMX500 format to be run on the AI camera. This segmentation will hopefully make it easier to manipulate the segmented area for easier QR decoding.

### 2/11/26  
precision_land_rc.py is the most up-to-date file. It uses a YOLO model run on the Raspberry Pi AI Camera to detect QR codes, calculate LANDING_TARGET parameters, and send the values to the Pixhawk flight controller
