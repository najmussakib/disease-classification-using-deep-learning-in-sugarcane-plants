# Disease-classification-using-deep-learning-in-sugarcane-plants
Crop diseases recognition is one of the considerable concerns faced by the agricultural industry. However, recent progress in visual computing with improved computational hardware has cleared the way for automated disease recognition. Results on publicly available datasets using Convolutional Neural Network (CNN) architectures have demonstrated its viability. To investigate how current state-of-the-art classification models would perform in uncontrolled conditions, as would be faced on-site, we acquired a dataset of five diseases of sugarcane plant taken from fields across different regions of Karnataka, India, captured by camera devices under different resolutions and lighting conditions. Models trained on our sugarcane dataset achieved top accuracy of 93.20% (on the test set) and 76.40% on images collected from different trusted online sources, demonstrating the robustness of this approach in identifying complex patterns and variations found in realistic scenarios. Taking everything into account, the approach of using CNNs on a considerably diverse dataset would pave the way for automated disease recognition systems.

## Dataset
The dataset contains 2940 images of sugarcane leaves belonging to 6 different classes (consisting of 5 diseases and 1 healthy). These include major diseases that affect the crop in India. All the images were taken in a natural environment with numerous variations. The images were taken at various cultivation fields including the University of Agricultural Sciences, Mandya Bangalore and nearby farms belonging to farmers.

![image](https://user-images.githubusercontent.com/19660042/188707514-913415ec-8895-483e-aee6-9152451c52d8.png)

## Approach 
- Used three popular architecture Resnet-50, Resnet-34, VGG-19 and assessed their applicability for the said classification problem.
