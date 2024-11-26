# **Project Overview**

This project explores an improved transfer learning scheme called intra-domain transfer learning, which focuses on transferring knowledge between datasets within the same domain rather than across different domains. This approach facilitates more effective knowledge transfer and improves model performance.

In this project, intra-domain transfer learning was applied to histopathology datasets. Models were pretrained on one histopathology dataset before being fine-tuned on another, enabling domain-specific feature learning and enhancing classification accuracy.

A total of four histopathology datasets were utilized from pretraining to target datasets.

# **Code Overview**

The Python scripts are named to indicate their specific functions. Below are a few examples:
* gashissdb pretrained.py - Pretrains models on the GasHisSDB dataset.
* gashissdb to chaoyang.py - Fine-tunes models pretrained on GasHisSDB to perform on the Chaoyang dataset.

# **Key Learning**

Through this project, I:
* Learned to pretrain models on domain-specific datasets instead of relying solely on ImageNet weights, leading to improved classification performance.
* Gained hands-on experience in implementing and fine-tuning transfer learning models using intra-domain datasets.
* Strengthened my understanding of advanced transfer learning techniques tailored to specific applications; for example, this project applied intra-domain transfer learning to histopathology image classification.
* Enhanced my coding skills in developing and optimizing deep learning models.
