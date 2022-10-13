# Automatic Number Plate Recognition - Nepali Number Plates

1. Description
2. Installation Guidelines
3. Testing & Evaluation
4. Techs used

## Description
The task of identifying a vehicle by extracting the vehicle license plate from an image or video is known as automatic number plate recognition. It reads symbols on license plates using optical character recognition. License plate localization on the image, character segmentation, character recognition of the segmented characters are the processes involved in a number plate recognition system. This research study aims to present an automatic Nepali number plate recognition system based on machine learning. Before the characters from a localized plate can be segmented and recognized, image artifacts must be removed from the image and the characters must be enhanced. This is done by applying various image pre-processing techniques to the localized plate image. The system then uses Convolutional Neural Network (CNN) for predicting each segmented character. The CNN is trained on a mixed Nepali character dataset comprising of characters generated from different computer fonts and handwritten characters. The accuracy of a such system depends significantly on many factors like image quality, illumination, skewness, distortions, and the presence of image artifacts.
[Full Report](https://github.com/praveshpansari/anpr-nepal/blob/main/report.pdf)

Preview:

![image](https://user-images.githubusercontent.com/25385289/195526910-855c15b2-e16f-4fd1-ad29-7a150b935042.png)



## Installation Guidelines

1.	Download the whole project folder and extract to preferred location
2.	Copy the folder ‘Web UI Flask’ from Implementation folder into a separate location
3.	This software requires Python 3.9 or above with the virtual env plugin installed. More libraries used are:
  a.	NumPy
  b.	Scikit Learn
  c.	SciPy
  d.	TensorFlow
  e.	Flask
  f.	OpenCV
  g.	Matplotlib
  h.	Pandas
4.	Download Python from Python.org, and save to a location
5.	Install Python by executing the file and choose required option and locations
6.	Install virtual environment using `pip install virtualenv`
7.	Activate virtual environment by executing activate in flask-app/Scripts from command line and install the libraries mentioned above
    `./flask-app/Scripts/activate`
8.	Run `py app.py`
9.	Go to http://127.0.0.1:5000 on a browser
 
 
## Testing & Evaluation
1.	Black Box Testing

  a.	Some characters are recognized correctly due to low quality image 
  
  ![half-correct](https://user-images.githubusercontent.com/25385289/195524124-6483d9b9-e675-46d2-9979-39eb691b7e67.png)

  b.	Full number plate recognized and labeled correctly
  
  ![correct](https://user-images.githubusercontent.com/25385289/195524157-f7800494-a19d-4246-9ea3-282a6365d9fa.png)

  c.	Black number plate recognized correctly
  
   ![black-correct](https://user-images.githubusercontent.com/25385289/195524169-0e936187-f809-45c3-a26e-f623ae88adc1.png)

2.	White box testing

a. Testing results of each class in character recognition

| Class 	| Precision (%) 	| Recall (%) 	| F1-Score (%) 	|
|-------	|---------------	|------------	|--------------	|
| 0     	| 100.00        	| 100.00     	| 100.00       	|
| 1     	| 97.18         	| 99.42      	| 98.29        	|
| 2     	| 98.04         	| 77.12      	| 86.33        	|
| 3     	| 97.78         	| 96.70      	| 97.24        	|
| 4     	| 99.03         	| 78.41      	| 87.52        	|
| 5     	| 95.57         	| 96.18      	| 95.87        	|
| 6     	| 96.63         	| 94.51      	| 95.56        	|
| 7     	| 91.29         	| 95.97      	| 93.57        	|
| 8     	| 98.50         	| 96.34      	| 97.41        	|
| 9     	| 93.38         	| 98.17      	| 95.71        	|
| MA    	| 90.18         	| 93.63      	| 91.87        	|
| KA    	| 89.34         	| 89.01      	| 89.17        	|
| SA    	| 98.30         	| 100.00     	| 99.14        	|
| JA    	| 93.77         	| 99.27      	| 96.44        	|
| NA    	| 89.60         	| 98.73      	| 93.94        	|
| BA    	| 98.88         	| 97.07      	| 97.97        	|
| GA    	| 78.74         	| 87.26      	| 82.78        	|
| LA    	| 86.55         	| 94.27      	| 90.24        	|
| DHHA  	| 97.50         	| 99.36      	| 98.42        	|
| BHA   	| 86.96         	| 88.24      	| 87.59        	|
| RA    	| 77.78         	| 93.96      	| 85.11        	|
| KHA   	| 87.58         	| 89.81      	| 88.68        	|
| DA    	| 80.31         	| 98.73      	| 88.57        	|
| YA    	| 100.00        	| 100.00     	| 100.00       	|
| GHA   	| 99.40         	| 95.38      	| 97.35        	|
| CHA   	| 94.51         	| 99.42      	| 96.90        	|
| JHA   	| 100.00        	| 99.42      	| 99.71        	|
| YNA   	| 100.00        	| 100.00     	| 100.00       	|
| THHA  	| 100.00        	| 94.22      	| 97.02        	|
| PA    	| 92.51         	| 100.00     	| 96.11        	|

b. Overall Accuracy, Precision, Recall, f1-score

Training Accuracy:	99.98%
Validation Accuracy:	94.11%
Precision:	94.44%
Recall:	95.02%
F1-Score:	94.15%


## Technologies Used:

![image](https://user-images.githubusercontent.com/25385289/195527452-2a29e4f7-b487-48a1-95b5-2f2fbced7f8f.png)


