# Gender-and-Age-detection-from-Voice
Gender and Age detection based on audio clips using GMM Model.

Dataset:
Source : Common Voice
  Audio Format: MP3
  
Data preprocessing: 
1. Extraction of unlabeled audio files from tsv file
2. Increasing time upto 10 seconds  of audio file by repeating the same, to make the training effective.

Gaussian Mixture Model:

GMMs are commonly used as a parametric model of the probability distribution of continuous measurements or features.

GMM parameters are estimated from training data using the iterative Expectation-Maximization (EM) algorithm or Maximum A Posteriori(MAP) estimation from a well-trained prior model.

We are extracting the train features of ‘male’ and ‘female’ recordings in gender and ‘young’, ‘adult’ and ‘middle age’ recordings in age and train the Gaussian Mixture Model.


