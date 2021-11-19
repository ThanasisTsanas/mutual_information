# mutual_information
Estimation of mutual information (vanilla method using density estimation approach)

This function computes the mutual information, which can be thought of as a more general method compared to correlation coefficients in order to quantify the statistical association between two random variables (vectors). Most freely available implementations of mutual information estimation rely on prior sub-optimal intermediate steps such as estimating probability densities using histogram techniques; here I provide a simple proof-of-concept approach estimating densities relying on kernel density estimation before computing the mutual information. Note there are more sophisticated and accurate approaches for computing the mutual information, but this (one might say naïve) implementation is simple, easy to understand, and computationally fairly efficient. The mutual information is not upper bounded which makes its direct interpretation difficult; for this reason I am also providing a normalised version. The normalised mutual information ranges between 0 and 1, where 0 denotes no association between the two random variables (that is, they are independent) and 1 denotes perfect association (knowledge of one random variable allows perfect prediction of the other). The function has been created with the standard goal in data analysis of determining the univariate association of each feature (attribute) with the outcome (target) we aim to predict.  

More details can be found in my PhD thesis, chapter 4:

A. Tsanas: Accurate telemonitoring of Parkinson’s disease symptom severity using nonlinear speech signal processing and statistical machine learning, D.Phil. thesis, Oxford Centre for Industrial and Applied Mathematics, University of Oxford, UK, 2012

[download from: https://www.dropbox.com/s/qnkfqmqonpvh9wi/DPhil%20thesis.pdf?dl=0]

****************************************
**% General call: [MI, MInormalized] = MI_ksdensity(X, y)**

%% Function to estimate the mutual information using kernel density estimation

Inputs:  
X        -> N by M matrix, N = observations, M = features

y        -> N by 1 vector with the numerical outputs

optional inputs:  

None

Output:  
 
MI      -> Mutual Information (computed using Parzen windows) each MI entry has the mutual information between the X(i) and y. (the MI units in this implementation are 'Nats') 

MInormalized -> Normalized Mutual Information values with respect to the mutual information of y. This enables direct comparison between the association strength of the features in X with respect to y

****************************************
Copyright (c) Athanasios Tsanas, 2014

**If you use this program please cite:**

A. Tsanas: Accurate telemonitoring of Parkinson's disease symptom severity using nonlinear speech signal processing and statistical machine learning, D.Phil. thesis, University of Oxford, UK, 2012

