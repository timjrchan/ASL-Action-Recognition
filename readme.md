<img src='http://imgur.com/1ZcRyrc.png' style='float: left; margin: 20px; height: 55px'>


# DSI-SG-42 Capstone Project:
### Silent Echoes: From Hand Waves to Written Phrases



<div >
    <img src="https://kidscarehomehealth.com/wp-content/uploads/2022/08/KCHH-sign-language-hello.jpg" alt="ASL image of signed Hello" style="width: 65%; height: auto;">
</div>


#  Introduction 

Across the globe, there are millions of people who uses sign language as their primary mode of communication in their daily lives. Whether the person is deaf, hard of hearing, or even mute, they use that to communicate between friends, family, and work environments. However, there is a communication gap between those that are unable to communicate by spoken language and those that do not understand sign languages. 

In the current world population, there are over [430 million people or over 5% of the world's population](https://www.who.int/news-room/fact-sheets/detail/deafness-and-hearing-loss) who have disabling hearing loss. [Hearing loss](https://www.who.int/news-room/fact-sheets/detail/deafness-and-hearing-loss) can be caused by a wide array of factors such as genetics, ear infections, some chronic diseases, and trauma to the ear or head. There are around 34 million children who are impacted by hearing loss and some would have to attend special schools to learn how to communicate with the general population. 

## Problem Statement

Creating an effective communication system between a deaf individual who uses sign language and a non-signing person presents significant challenges. The initial goal of this project is to lay the foundation for such a system by developing a solution that can accurately recognize and translate a select set of key signs - currently three critical words - into written language.


**This project aims to demonstrate the feasibility of scaling this technology to enable more comprehensive interactions and foster understanding between signing and non-signing individuals.**


## Aims

The aim of this project is to bridge the gap by using data science to help the general public to understand sign languages so that the *signer* can convey their message naturally and fluidly without losing any of the nuances in body language. 


### [HandSpeak](https://www.handspeak.com/word/most-used/)

This website recommends a list of words that a peson who wants to learn sign language. It is split into blocks of 100 words. We will be focusing on the first page with the most common sign languages. 


This website offers a structured learning resource for aspiring sign language learners. The resource takes the form of a categorized vocabulary list of one hundred words each. Our initial exploration will be based on the words presented on the first page, focusing on the most frequently utilized words in everyday communication. These signs will encompass pleasantries, question words, and common everyday used words. The website structures the words in alphabetical order rather than learning order, we will shortlist 10 words from the list to build the initial model.

*n.b. as the video files and extracted landmark .npy files are too big for GitHub, they can be found on my [Google Drive](https://drive.google.com/drive/folders/1FcOdltQ70QdpTyonlIWNVPSipTXkLZ6m?usp=sharing)*


# Dataset

Our data comes from two different datasets that comprises of videos for American Sign Language. 

1. [Microsoft American Sign Language](https://www.microsoft.com/en-us/research/project/ms-asl/downloads/): 
        This dataset has over 25,000 annotated videos within the datasets which has been split into `train`, `test`, and `validation` dataframes. Each of the datasets has:
        - the name of the sign/word class
        - the name of the file
        - start and end frames/time
        - the dimensions of the video file
        - the frames per second (fps) of the video 
        - URL of the video
        - boundary box that has been normalized to the dimensions of the video
        - the signer id
<br/>

2. [World Language American Sign Language](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed):
    This dataset is found on Kaggle and at the point of submission has 2,000 common different words in ASL. It is a nested json file that contains the following:
    - the sign/word class (gloss)
    - the name of the video file
    - the start and end frames
    - the source of the video file
    - the URL of the source video
    - the frames per second (fps)
    - the signer id


# Data Dictionary
| Column              | Data Type | Dataset      | Description                                                  |
|---------------------|-----------|--------------|--------------------------------------------------------------|
| finger_tip_y        | float     | merged_df.csv | The y coordinate of the respective finger tip in the frame of the video |
| thumb_tip_y         | float     | merged_df.csv | Tip of the thumb                                             |
| index_finger_tip_y | float     | merged_df.csv | Tip of the index finger                                      |
| middle_finger_y    | float     | merged_df.csv | Tip of the middle finger                                     |
| ring_finger_tip_y  | float     | merged_df.csv | Tip of the ring finger                                       |
| pinky_finger_tip_y | float     | merged_df.csv | Tip of the pinky finger                                      |
| please              | object    | merged_df.csv | The word class used for the signing                          |
| sorry               | object    | merged_df.csv | The word class used for the signing                          |
| hello               | object    | merged_df.csv | The word class used for the signing                          |

<br/>

# Workflow Process

The word classes were decided with some research on the web where the top 10 words can be used for classification. However, due to the limitations of the video files available, we further reduced the number of word classes to three to prevent any class imbalance. The signed words decided were `please`, `sorry`, `hello`. These are chosen as it is commonly used words in English and can be learned relatively  quickly. 



- **Please**: In ASL, `'please'` is often expressed by placing the palm facing down, and moving it in a circular motion over your chest in a clockwise direction. This sign is often accompanied by a pleading facial expression to convey politeness and sincerity.

<img src = 'https://res.cloudinary.com/spiralyze/image/upload/f_auto,w_1414/BabySignLanguage/DictionaryPages/please-webp.webp' alt = 'an illustration of signing please' style='width: 50%; height: auto;'> 
<br/>


- **Sorry**: The sign for `'sorry'` in ASL involves placing one hand over your heart, fingers together and pointing downward, and making a small circular motion. This sign is typically accompanied by a remorseful facial expression to convey genuine regret or apology.

<img src = 'https://res.cloudinary.com/spiralyze/image/upload/f_auto,w_1414/BabySignLanguage/DictionaryPages/sorry-webp.webp' alt = 'an illustration of signing sorry' style='width: 50%; height: auto;'>
<br/>


- **Hello**: To say `'hello'` in ASL, you can use a simple wave gesture. Extend your dominant hand with fingers spread apart and move it slightly side to side in front of your body. You can also combine this with a friendly facial expression to greet someone warmly.

<img src = 'https://res.cloudinary.com/spiralyze/image/upload/f_auto,w_1414/BabySignLanguage/DictionaryPages/hello-webp.webp' alt = 'an illustration of signing hello' style='width: 50%; height: auto;'>
<br/>

####  Downloading of videos
The videos from `Microsoft-ASL` dataset are required to be downloaded from YouTube which the `pytube` library allows us to download seamlessly. 

#### Cleaning of videos
The raw video files are of different durations which some are a compilation of signed words. In this project, only the `Microsoft-ASL` dataset requires video cropping according to the start and end frames as provided. We have tried to crop the videos with the start and stop time but that has caused video stuttering. Hence, the start and end frames were used instead.

The `World Langauge - ASL` dataset has the videos already cropped. 

The videos from both dataset has different video dimensions and fps values. The model requires the video dimensions and fps to be consistent. Thus, the videos from both datasets have been resized to `512 x 512` pixels at `30 fps`. These dimensions chosen are a middle ground to the varying resolutions in the video files while most of the videos are 29 fps. One of the goals of this project is to use a camera for real-time prediction and webcams are able to operate at 30 or 60 fps. As such, we have chosen 30 fps as it is most appropriate. 

### Augmenting Videos

The sample sizes for the videos are similar but considered few for model training - `please`: 30, `sorry`: 18, `hello`: 17. Augmenting videos allow us to create additional sample videos while creating different real-world scenarios, allowing a more robust model. These are the augmentation techniques used in this project:

&emsp;- flipping horizontally
&emsp;- reducing the brightness
&emsp;- increase the contrast
&emsp;- decrease the contrast
&emsp;- shearing left
&emsp;- shearing right
&emsp;- gaussian blur

##### Flipping Horizontally 
Horizontally flipping the videos increases the diversity of the training data by effectively doubling the amount of data available without changing the semantics of the video. It also encourages the model to learn features from another viewpoint, like in signed actions, flipping the video essentially changes the signing hand. This would help improve its generalization ability. 

##### Reducing the Brightness
This simulates scenarios with varying lighting conditions, making the model robust to changes in illumination, it also prevents the model from overfitting to a specific lighting conditions present in the training data. Most of the videos in the dataset are in controlled environment with adequate lighting. 

##### Adjusting the Contrast
Changing the contrast can enhance or diminish the differences in intensity between pixels, which can bring out or supress certain features in a video. This can make the model more resilien to variations in contrast that may occur in the real-world scenarios.

##### Shearing Left and Right
Shearing left or right introduces distortion that simulates the effects of perspective changes or camera movements, making the model more robust to such variations. This encourages the model to learn the signed words from different angles or viewpoints, improving its ability to recognize objects from diverse perspectives.

##### Applying Gaussian Blur
Gaussian blur smooths out fine details in the video, which can help the model focus on more prominent features and reduce sensitivity to noise. This would mimic the effect of motion blur that may occur due to camera or the signer's motion, making the model more robust.


## Exploratory Data Analysis

We use varying complexity with `MediaPipe` library to superimpose a skeletal view on the model to visually observe how it detects the person signing. 

The landmarks from the hands are extracted and the coordinates of the right-hand finger tips are used to plot a line graph. Most of the signers are right-handed and the finger tips provides more insight to how it is viewed within the frame. 

Cluster analysis is conducted to explore if the word class can be clustered and can be classified distinctly from each other. 

A Principal Component Analysis (PCA) is conducted as a final step to investigate further and reduce the dimensionality of all five fingers if it can be explained by just two dimensions.

### Insights from EDA

We have found trends and patterns from each of the sign actions where `hello` has a 2 step motion. The signing actions of both `please` and `sorry` shows 'wavy' pattern in the plot but the closeness of the finger tips differ in how tight the lines are.

Cluster analysis shows distinctive clusterings between the word classes with `hello` usually found higher up in the video frame.

PCA shows that 99.5% of the variation can be explained with the first axis further reinforcing that the word classes are different from each other.

## Modeling

The final model architecture selected using Long-Short Term Memory (LSTM) Model as the first (input) layer as the frames in the video are sequential and depends on the previous frame. The second (hidden) layer allows additional training with the same number of neurons (`64`) as the first LSTM layer. An dropout layer (`0.5`) is added to prevent overfitting to this layer. The final (output) layer has 3 neurons with `softmax` activation which represents the probability of each word classes. The model is fit with the training data and validation data so the loss and accuracy can be obtained after the training. 

#### Results after modeling
After modeling the 3 different model complexities, we have the following results:

|                   | Accuracy | Precision | Recall | F1-Score |
|-------------------|----------|-----------|--------|----------|
| Comprehensive Model |   0.69   |    0.73   |  0.69  |   0.68   |
| PH Model          |   0.2    |    0.37   |  0.2   |   0.14   |
| Hands Model       |   0.87   |    0.88   |  0.87  |   0.84   |


The best performing model is the Hands Model with an accuracy of 0.84 while the worse perfoming model is the PH Model with an accuracy of 0.2.


## Real-Time Prediction
We tested the model in real-time with my laptop's webcam and found that that model can predict the different actions for ASL.

## Conclusions
From the modeling, we proved that using action signs for a word is plausible even with a small sample size. Moreover, the data extracted through `MediaPipe` shows graphically that each signing word has different patterns and can be classified distinctively.


## Limitations
Currently, this model is limited by the small sample size and may not be robust enough in multiple scenarios as the training videos are taken with a plain background with only a person in frame. The model is also limited by the small word class that it is able to detect. To contextualize this model to Singapore's environment, we will need to train the model to fit Singapore's Sign Language instead of using the American Sign Language.

## Further Works
We intend to expand this model's ability to be used in Singapore and would like to work with the relevant parties involved with the hearing impaired to localise the competency of this model and expand the word classes to predict or classify more signing words. 