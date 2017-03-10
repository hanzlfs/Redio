# REDIO
Recognition Events of auDIO: AI project with Insight AI

### Cobalt Robotics | Audio Classification
### Problem 
Cobalt Robotics | Audio Classification | Problem description
Our robot is equipped with a sensitive microphone for hearing things around it, and we would like to classify a multitude of different sounds that the robot might encounter around it. There are several online databases with large numbers of sound effect samples, such as http://www.freesound.org/browse/tags/window/. Some initial topics that we’d like to classify are glass breaking, people talking, footsteps, car horns, car noises, music, and fans. 

##### Description of the data 
There should be enough online databases and sources of pre-classified data for the Fellow to collect themselves. 

##### Prior Work 
Previous initial work done to classify general audio anomalies, including doing some feature extraction from audio data. 

##### Deliverable 
We would like a model that can classify a 1 second audio clip in real time. Our robot runs a full linux desktop environment with a GPU. Implementations in python would be easiest to integrate. We would also like the full training pipeline so that we can retrain the model as we collect more data. There are some prior works doing music classification in Keras: https://keras.io/applications/#musictaggercrnn. 

How Cobalt Robotics would implement the result If we agree that the model is reasonable we will integrate it into our code and deploy it onto all of our robots.

