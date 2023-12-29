# Multimodal-learning-performance-using-both-VR-derived-and-MRI-biomarkers
This repository contains the Python code used for multimodal learning, aiming to enhance the performance of early Mild Cognitive Impairment (MCI) detection by combining statistically significant Virtual Reality (VR)-derived and Magnetic Resonance Imaging (MRI) biomarkers. The chosen machine learning model is the Support Vector Machine (SVM) algorithm, known for its effectiveness in similar tasks.

## VR-derived biomarkers
![Figure3_Extraction of four VR biomarkers](https://github.com/seoultech-HAILAB/Multimodal-learning-performance-using-both-VR-derived-and-MRI-biomarkers/assets/125949680/f933a3bc-25f3-4112-9d2c-36d9a38e47ee)
Extraction of four VR-derived biomarkers from behavioral data in the virtual kiosk test. Hand movement speed is calculated using the collected hand movement data from the virtual kiosk test. Scanpath length is derived from the eye movement data. The time to completion and the number of errors are calculated based on the performance data.

## MRI biomarkers
![Figure4_Extraction of 16 MRI biomarkers](https://github.com/seoultech-HAILAB/Multimodal-learning-performance-using-both-VR-derived-and-MRI-biomarkers/assets/125949680/642d8ace-6c9e-4a2a-b5d6-e4ab77987833)
Extraction of 22 MRI biomarkers from both hemispheres using the Split-attention U-Net architecture. Following multi-label segmentation of the brain’s region of interest, each brain region’s volumes are quantified as MRI biomarkers. Each hemisphere has eleven biomarkers including the cerebral white matter, cerebral gray matter, ventricles, amygdala, hippocampus, entorhinal cortex, parahippocampal gyrus, fusiform gyrus, and superior, middle, and inferior temporal gyrus. 
