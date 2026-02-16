# What Is Statistical Machine Learning?

This course, "Statistical Machine Learning" (SML), builds directly upon the foundational knowledge you have gained in introductory machine learning courses such as [**FDS**](https://fum-cs.github.io/fds/) and [**A4DS**](https://fum-cs.github.io/a4ds/). While those courses introduced you to the landscape of machine learning algorithms and their application, this course takes a deeper dive into the *probabilistic foundations and statistical principles* that underpin them.

We will move beyond viewing machine learning as a mere set of tools for prediction and start treating it as a framework for **statistical inference** and **data modeling**. Our goal is to understand not just *how* algorithms work, but *why* they work, and under what assumptions they are optimal.

## From Algorithms to Models: A Probabilistic Perspective

In your previous courses, you learned to categorize machine learning problems into supervised (classification, regression) and unsupervised (clustering, dimensionality reduction) learning. You learned that "learning" involves adjusting tunable parameters to fit observed data. In this course, we will formalize this process through the lens of probability theory and mathematical statistics.

We will reframe these problems as exercises in statistical modeling. For instance:

- **Supervised Learning** can be seen as modeling the conditional probability distribution of the labels given the features, $P(\text{label} | \text{features})$.
- **Unsupervised Learning** can be seen as modeling the probability distribution of the features themselves, $P(\text{features})$.

This probabilistic view provides a unified and principled way to approach model fitting, evaluation, and interpretation. It allows us to quantify uncertainty, make optimal decisions, and understand the fundamental limits of learning from data.

## The Core Pillars of Statistical Machine Learning

This course is structured around the core concepts that form the bedrock of statistical machine learning. We will explore these topics in depth, building a coherent mathematical framework.

### 1. Probabilistic Foundations and Parameter Estimation

Before we can build complex models, we must establish the fundamental tools for describing and understanding data from a statistical perspective.

- **Understanding the Covariance Matrix:** We will move beyond a simple definition and explore the covariance matrix as a geometric transformation that describes the shape, scale, and orientation of data. This understanding is crucial for methods like Mahalanobis distance, Principal Component Analysis (PCA), and Gaussian Mixture Models (GMMs).
- **Maximum Likelihood Estimation (MLE):** This is perhaps the most fundamental concept in statistical learning. We will formally introduce MLE as a general principle for estimating the parameters of a probability distribution given observed data. You will see how many familiar algorithms, from linear regression to Naive Bayes, can be derived from this single, powerful idea.
- **Mahalanobis Distance:** Building on the covariance matrix, we will learn about the Mahalanobis distance, a distance metric that accounts for the correlations and scales of the data. It is a key component in many classification and clustering algorithms and provides a more meaningful measure of similarity than Euclidean distance in many real-world scenarios.

### 2. Bayesian Learning and Decision Theory

The Bayesian paradigm offers a powerful alternative to the frequentist approach (like MLE) by incorporating prior beliefs and quantifying uncertainty in a principled way.

- **Bayesian Classification:** We will move from the simple, often Gaussian-based Naive Bayes classifier to a full Bayesian treatment. This involves specifying prior distributions over model parameters and updating these beliefs with data to obtain posterior distributions.
- **Bayesian Decision Theory:** This provides a formal framework for making optimal decisions in the presence of uncertainty. We will learn how to use probabilistic models (like posterior distributions) and a loss function to make decisions that minimize expected risk. This connects probabilistic modeling directly to practical action.

### 3. Linear and Nonlinear Modeling

We will revisit linear models, not just as algorithms, but as interpretable statistical models with well-understood properties. From there, we will explore the elegant mechanism of the kernel trick to extend these linear models to handle complex, nonlinear relationships.

- **Linear Models for Regression and Classification:** We will study models like Linear Regression and Logistic Regression in detail, examining their assumptions, derivations (often via MLE), and extensions.
- **Orthogonal Matching Pursuit (OMP):** As a bridge to more advanced topics, we will explore OMP as a method for sparse recovery. This connects linear modeling to the growing field of sparse optimization, which is vital for high-dimensional data.
- **The Kernel Method (Kernel Trick):** This is a cornerstone of modern machine learning. We will dissect the kernel trick, understanding how it allows us to implicitly map data into a high-dimensional feature space and apply linear models without ever computing the coordinates of the data in that space. This will provide the foundation for understanding kernelized versions of algorithms like Ridge Regression, PCA, and clustering.
- **Kernel K-means Clustering & Kernel Regression:** We will apply the kernel trick to concrete examples, demonstrating how it can transform linear algorithms into powerful nonlinear ones, such as Kernel Regression, which can model complex functions.

### 4. Generative Models for Unsupervised Learning

We will delve into sophisticated models for uncovering hidden structure in unlabeled data.

- **Gaussian Mixture Models (GMMs):** We will dedicate two parts to this important topic. GMMs are a powerful probabilistic clustering method that models data as a mixture of several Gaussian distributions. We will learn about the Expectation-Maximization (EM) algorithm, a general technique for parameter estimation in models with latent variables, and see how it is used to fit GMMs.
- **Principal Component Analysis (PCA):** We will cover PCA in two parts, developing a deep understanding of it from multiple perspectives. We will derive PCA as:
    1.  Finding the directions of maximum variance in the data.
    2.  Finding the low-dimensional projection that minimizes reconstruction error.
    We will explore its connection to the covariance matrix and the singular value decomposition (SVD), solidifying it as a fundamental tool for dimensionality reduction and feature extraction.

### Summary and the Road Ahead

In summary, "Statistical Machine Learning" is not just a collection of new algorithms. It is a shift in perspective. It is about building a rigorous mathematical and probabilistic foundation that allows you to:

- **Understand** the principles behind a wide array of machine learning methods.
- **Derive** new algorithms from first principles.
- **Quantify** the uncertainty in your model's predictions.
- **Make optimal decisions** based on data.
- **Critically evaluate** the assumptions and limitations of your models.

The concepts covered in this introductory chapter—the covariance matrix, MLE, Mahalanobis distance—are the very building blocks for everything that follows. Mastering them is the first step on the path to becoming a thoughtful and effective practitioner of statistical machine learning.

#### Some of my previous papers machine learning:

- Classification:    
    * Click Rate Prediction {cite}`Fatehinia1403dsas3`
    * Anti-Cancer Plant Recommendation {cite}`Amintoosi2023GFS-COAM`
    * Fully Connected to Fully Convolutional {cite}`FC2FC_2022`
    * Eigenbackground {cite}`JMIV-2022`
    * Facial Recognition {cite}`Sadoghi07facial`
    * COVID 19 {cite}`Reg-OBD-Corona-Detection`
    * Social Networks {cite}`GCN-JAC2021`
    * Classification of Paintings {cite}`TaylorExpansion_in_CNN_prunning99`
- Regression: 
    * Modeling dust particles {cite}`Zolfaghari2025Modeling`
    * Housing Price Prediction {cite}`Amintoosi1403dsas3`
    * Traffic prediction {cite}`GSL-TF1403aimc55,Bagherpour1403dsas3`
    * Sparse Super Resolution {cite}`Visual-2022`
    * Prediction of CO and PM10 {cite}`Farhadi2022`
    * Prediction of the Air Quality {cite}`Farhadi2019`
- Classification & Regression:
    * Fire detection {cite}`ST_for_DA_2022`
    * Predicting Molecular Properties {cite}`Jlogir1403dsas3`
- Semi-supervised learning:
    * Text extraction {cite}`Nemati2024dsas3`
    *  Vessel Segmentation {cite}`Amintoosi94matting`
- Un-supervised learning:
    * DeepWalk for Student Sectioning {cite}`Amintoosi2024-Deepwalk-SSP`
    * Overlapping Clusters {cite}`GCN-JAC2021`
    * Min-Cut of Weighted Graphs {cite}`Hosseini98-TSCO-Karger`
    * Spectral Clustering {cite}`Nemati96GA`
    * Retina Vessel Segmentation {cite}`Amintoosi96IPRIA-Matting`
    * Extreme Learning Machine {cite}`Amintoosi96IPRIA-ELM`
    * MRI Images Segmentation {cite}`Amintoosi94spectral`
    * Graph Minimum Cut Using SA & TS {cite}`Hoseini93mincutSA,Hoseini93mincutTS`
    * Pulse-Coupled Neural Networks {cite}`Moghimi92MRIPCNN`
    * Segmentation of Medical Images {cite}`Sheida90Unified`
    * Fish School Clustering {cite}`Amintoosi07afishschool`
    * Fuzzy Student Sectioning {cite}`Amintoosi05feature,Amintoosi04fuzzy`
