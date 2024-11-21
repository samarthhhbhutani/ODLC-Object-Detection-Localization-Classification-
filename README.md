# Object Detection Localisation and Classification

This repository contains the code for an autonomous aerial system that detects and reads text on ground targets using an in-house Region Proposal Network (RPN), geotagging, and Optical Character Recognition (OCR).

It utilized Wide ResNet-50 for Optical Character Recognition and Detecting colour and YOLO V8 for Region Proposal.

Directory Structure:
Final-PipeLine-Run: Contains the final pipeline integrating all parts (ODLC+DroneKit+Takeoff+Land)
ODLC-Sub-PipeLine: Contains the pipeline integrating ODLC
Waypoints.txt: Contains the waypoints to which drone is to be sent to
