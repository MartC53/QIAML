# Users Stories 
## Assay developers:
Advanced user that uses and modifies the software to apply the workflow to develop new assays for different diseases. This user applies their sample and standards to produce a new model for quantification of new diseases. Ideally interacts with the software at a deeper level (command line) to alter the parameters to match their setup (focal length, target recognition, etc.). This user has fundamental assay knowledge to allow them to produce the imaging prototype box, understands the fluorescence readout mechanism (what fluorescent channels are being utilized). 
- Produce new standard curves
  - Fundamentally change the ML model
- Produce Experimental imaging set up
  - Alter the image color channel for different molecular probes

## Lab bench user:
More advanced user that makes edits or is trying to implement the software for the purpose of subtype or variant detection of existing models. This user has major interaction with the software and has background knowledge in python. This user is the primary user until “at home” version is created. Ideally interacts with the software from a cloned repo or website. Compared to the assay developer, the lab bench user is an optimizer or a new project member who is trying to extend the model rather than create a new model.
- Can upload their data and make predictions based on a pre-defined model.
- Can alter image pre-processing steps to increase image clarity.

## Clinicians:
Demonstrates point of care use case, takes the samples from the patient, and runs a diagnostic test and reports results to the patient. This user would be the primary user for an at home/ in clinic software planned for future development. This user has limited to no background knowledge of jupyter notebooks/python.
- Images samples 
- Ideally interacts with a web interface or app to load in images 
- reads out results 

## Programmer:
Trains the machine learning algorithm for software. Programmers need to get access to many sample images, run the codes for analysis and compare the results with PCR data. The images the patient got from the cellphone should be locally processed. 
- Fundamentally changes model 
- Validates the model by comparing clinical samples to test data sets

## Patient:
Provides sample and then receives results in further versions will be the only user that interacts with the application for “at home” use. Limited or no background knowledge of jupyter notebooks/python.
- Images samples 
- Ideally interacts with a web interface or app to load in images 
- reads out results 

# USE CASES
## User inputs:
- Select operation create new quantification / Upload New standard / Change assay (disease)
- Feeding Images
- Post quantification actions: save to device, send to doctor, call healthcare.
- Trends in history

## GUI- Interaction with 1. Website 2. App (Future) 
- Progress bar
   - Text Return for progress 
- Return Number / Compared to standard
   - Graph of counts, visual display of counts on standard curve
   - Output to show accuracy of each image with error bar
   
## Components 

```import_data```

What it does: First step to allow users to load their data. Takes in a directory of image location. Then applies ```autocrop``` functions that crop the images down to the membrane. functions also blur images and apply a contrasting algorithm to aid in visualization.
Inputs: User acquired RGB jpg images.
Outputs: .npy files containing the image data flattened to grayscale and cropped to the membrane.
Interactions: User calls ```import_data``` function at a script level. Ideally users would upload images to website, and this would be the first part of the workflow.

```get_data```

What it does: Allows for simple loading of datasets created by the ```import_data``` functions.
Inputs: Named dataset created in ```import_data```
Outputs: NumPy arrays or Pandas dataframes preprocessed and ready for the ML model
Interactions: User calls ```get_data``` function at a script level. Ideally would automatically follow ```import_data``` in a GUI interface.

```ML model```

What it does: Regression analysis of Image to make a quantification estimate 
Inputs: Takes preprocessed data from ```get_data```
Outputs: A trained regression model that has quantification prediction abilities
Interactions: User interacts at a script level for new assays. For developed assays users could load results from pre-trained models to apply to their data. Unlikely a GUI would have training abilities- user would load pretrained model.

```result_overview```

What it does: shows user results- sanity check by showing user image of amplification pad along with estimate number of copies. May overlay previously developed region spot analysis code for tertiary evaluation.
Inputs: none
Outputs: image of amplification pad and quantification estimate.
Interactions: User interacts with this as a pop-up window or GUI on website.

   
## Process flow
1. Images collected on smartphone
    - Automated app based on android APK exists allowing for complete camera control (focal length, iso, flash time, exposure time, time between frames)
      - Built on python for android with kivy, considerations may need to be realized to allow integration with python for android (no scikit learn, yes open cv)
2. Image preprocessing 
  - Evaluate effect on ML with existing techniques
    - cropping
    - adaptive contrasting 
    - blurring
    - adaptive thresholding
3. Train/ test iterations of ML models on smartphone .jpg images
    - Due to small dataset image augmentation may be needed
    - Microscope images could be used as supplement, check for bias
      - Microscope images are .tiff and much larger
4. Display results as numerical answer (# copies) from y_predict
5. Display results on a standard curve that is annotated with clinical decision-making matrix
6. Display results with overlayed spot counter as sanity check.
