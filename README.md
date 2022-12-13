# Emotion_recognition_and_speakout(computer vision)

What does this project do?
1. This shows the emotions of every student.
2. Recognizes faces.
3. Generates a file every day after recognizing faces.
4. Speak out wishes for the higher officials when they enter.

======================================================================================================

 Tools used (requirements):
 1.TensorFlow
 2.keras
 3.OpenCV
 4.CUDA
 
 =====================================================================================================
 
Warning*
Code will only work when you set the appropriate parameters according with your dataset.
For better acuracy use learning rate = 0.001
epochs = 50 (in your dataset is in thousands)
steps_per_epoch = data points // batch_size
batch_size = between 16 - 64
I used target_size = 48 x 48 (you can change it accordingly by your dataset)
 
 =====================================================================================================
 
we are import Dispatch from win32com.client to speakout when we found a match or when it recognizes.
This can detect multiple faces from a frame.

 
