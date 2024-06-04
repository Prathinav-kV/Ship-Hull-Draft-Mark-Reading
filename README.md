# Ship-Hull-Draft-Mark-Reading
Computer Vision Project to read the Ship Hull Draft Marking to estimate the water level
# Water Level Detection using Optical Flow and Pytesseract

## Project Overview

This project focuses on detecting water levels from video footage of a ship hull using optical flow methods and Pytesseract for text recognition. The aim is to enhance the accuracy and efficiency of water level detection, which is crucial for various maritime applications.

## Team Members

- Prathinav Karnala Venkata
  - M.Eng Robotics, University of Maryland, College Park, Maryland, USA
  - Email: pratkv@umd.edu

- Pranav ANV
  - M.Eng Robotics, University of Maryland, College Park, Maryland, USA
  - Email: anvpran@umd.edu

- Sarang Shibu
  - Robotics Program, University of Maryland, College Park, Maryland, USA
  - Email: sarang@umd.edu

## Key Contributions

- **Optical Flow Method**: Utilized the Farneback method for optical flow to track water levels on a ship hull.
- **Pytesseract Integration**: Implemented Pytesseract for OCR to read text from the video frames.

## Results

- **Performance**: Conducted multiple test cases demonstrating the efficiency and effectiveness of the water level detection method.

## Methodology

The project implemented the Farneback method for optical flow to track the movement of water levels in video frames. Pytesseract was used for OCR to extract textual information from the video, enhancing the detection accuracy.

## Applications

The findings of this project are applicable to real-world maritime scenarios where accurate and efficient water level detection is crucial.

## Conclusion

The project successfully demonstrated the potential of combining optical flow and OCR for water level detection, identifying challenges such as parameter sensitivity, handling varying lighting conditions, and computational efficiency.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- Pytesseract

How to execute: 

1. Install Pytesseract from this link: https://github.com/UB-Mannheim/tesseract/wiki

2. Within the code add the directory to the tesseract.exe file to the tesseract_cmd variable within the project5_final.py file

3. Download the 'hull.mp4' file and place it in the same workspace as the .py file.

4. Execute project5_final.py in Visual Studio

5. Final Video will be generated within the workspace after code has been successfully executed.
