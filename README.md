# Unmasked-An-in-depth-analysis-of-deepfake-detection-models

![image](https://github.com/user-attachments/assets/a5adbe53-ee6a-4b10-855b-00dd66e04679)

## Introduction

This project implements and evaluates the Uncovering Common Features (UCF) Spatial model for deepfake detection. The model is trained on the FaceForensics++ dataset and tested across WildDeepfake and CelebDF datasets to assess cross-dataset generalization. The model’s performance was optimized through extensive hyperparameter tuning (e.g., batch size, learning rate, noise), and its interpretability was enhanced using tools like TensorBoard, Grad-CAM, and t-SNE.

Additionally, a fairness and bias analysis was conducted, leveraging demographic annotations from the datasets to evaluate the model’s performance across diverse groups. The goal is to develop robust and ethical deepfake detection systems by addressing both technical and societal challenges.

This work provides valuable insights into deepfake detection, with a strong focus on both performance and fairness, while offering an open-source framework for future research.

## Datasets
### 1. FaceForensics++
FaceForensics++ contains 1000 original videos altered with four face manipulation techniques: Deepfakes, Face2Face, FaceSwap, and NeuralTextures. The dataset includes 977 YouTube videos featuring clear, unobstructed faces, enabling realistic deepfake creation. It is equipped with binary masks suitable for tasks like classification and segmentation, along with 1000 pre-trained Deepfakes models for augmenting data.

### 2. WildDeepFake
WildDeepfake consists of 7,314 face sequences from 707 deepfake videos sourced from the internet. While smaller than other datasets, it provides real-world deepfake footage, offering a challenging environment for deepfake detection. This dataset helps test detection models against real-world scenarios where performance tends to drop significantly.

### 3. Celeb-DF
Celeb-DF contains high-quality deepfake videos featuring celebrities, generated with advanced manipulation techniques. This dataset evaluates deepfake detection models in diverse real-world conditions. It is used to test the generalization capability of the UCF Spatial model on more challenging deepfake data.

### 4. AI-Face
The AI-Face-FairnessBench dataset includes facial images annotated with demographic attributes such as age, gender, and skin tone. It allows fairness analysis for facial recognition and deepfake detection models. This dataset ensures that our model performs equitably across diverse demographic groups and helps identify potential biases.

## Preliminary Results
Comparing our results obtained using the pretrained weights with the author’s results for the entire FaceForensics++ and WDF datasets, we observed that the results were promising, yielding an accuracy of 0.88 and an Area Under the Curve (AUC) of 0.94. While these metrics indicate strong performance, they are slightly lower than the benchmarks established by the original authors, who reported an AUC of 0.96 using the FF++. Considering the WildDeepFake dataset’s known difficulty due to its diverse and sophisticated deepfake content, our assumptions were validated when we observed a decrease in the model’s performance when it yielded an accuracy of 0.63 and an Area Under the Curve (AUC) of 0.73. These results indicate that the model faces significant challenges in handling the complexity and variability of this dataset.

<img width="283" alt="Screenshot 2024-12-18 at 7 08 55 PM" src="https://github.com/user-attachments/assets/2e2b7801-070a-4bf4-9fc7-7d2417914938" />

