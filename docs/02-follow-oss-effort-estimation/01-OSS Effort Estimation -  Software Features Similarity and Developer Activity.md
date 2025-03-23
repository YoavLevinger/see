### Briefing Document: OSS Effort Estimation Using Software Features Similarity and Developer Activity-Based Metrics
[Ritu Kapur and Balwinder Sodhi. 2022. OSS Effort Estimation Using Software Features Similarity and Devel-
oper Activity-Based Metrics. ACM Trans. Softw. Eng. Methodol. 31, 2, Article 33 (February 2022), 35 pages.]

### 1. Overview

This research paper presents a novel method for estimating the effort required to develop software, particularly within the realm of Open Source Software (OSS). The core concept is to leverage developer activity information from past software projects, combined with a similarity detection model, to predict the effort needed for a new, envisioned software project. The authors are R. Kapur and B. Sodhi, from the Department of Computer Science and Engineering, Indian Institute of Technology Ropar, India.

### 2. Key Themes and Ideas

Software Development Effort Estimation (SDEE): The paper addresses the challenge of accurately predicting the effort (e.g., person-months) needed to develop software. It acknowledges existing SDEE methods (expert-based, algorithmic, and hybrid) but aims to improve accuracy by incorporating developer activity and software similarity.
Developer Activity Information: A central tenet of the research is that metrics derived from developer activity within OSS repositories provide valuable insights into development effort. Specifically, the paper defines developer activity information as:
"(a) Source code contribution by a developer : The number of additions, deletions, and modifications of source lines of code (SLOC) performed by a developer on a software."
"(b) The timeline of various changes in a VCS: This acts as a valuable metric for computing the time required to develop a software product."
"(c) The metadata information of the software repository: Information such as the number of developers working on the software repository is essential for computing the effort required to develop the software product."
Software Similarity Detection: The paper uses the Paragraph Vector algorithm (PVA) to determine the functional similarity between software projects based on their descriptions. This allows the system to identify past projects on GitHub that are similar to the newly envisioned software.
Definition 2: Paragraph Vector algorithm "The Paragraph Vector algorithm (PVA) is an unsupervised machine learning (ML) algorithm that learns fixed-length feature representations from variable-length pieces of texts, such as sentences, paragraphs, and documents. The algorithm represents each document by a dense vector trained to predict words in the document [Le and Mikolov 2014]."
SDEE Tool: The research culminates in the development of an SDEE tool that combines the SDEE dataset (derived from GitHub) and the software similarity detection model. The tool takes a description of software requirements as input and predicts the required development effort.
SDEE Metrics: The paper proposes new SDEE metrics based on developer activity information such as:
|Dr|: The total number of developers involved.
Tr: The total time spent in developing the software (calculated based on release start and end times).
er: The effort expended to develop the software, calculated as the product of the number of developers and the development time (er = |Dr| ∗ tr).
GitHub as a Data Source: The research relies heavily on GitHub as a source of data for developer activity and software project descriptions. The authors systematically downloaded and processed information from GitHub repositories belonging to specific software categories.

### 3. Methodology and Approach

The paper outlines a multi-step approach:

Dataset Development:Processing GitHub software repositories and developing an SDEE dataset based on developer activity information.
Choosing software categorizations from MavenCentral to ensure homogeneity.
Downloading software project descriptions from corresponding GitHub repositories.
Similarity Model Training:Training a software similarity detection model using the PVA on the project description documents from the dataset.
SDEE Tool Development:Designing a GUI for user interaction.
Computing effort estimates for a newly envisioned software project by leveraging the SDEE dataset and PVA vectors.

### 4. Design Considerations

The authors address several important design decisions in their approach:

Homogeneity of Dataset: Ensuring that the dataset contains a diverse yet representative range of software types.
GitHub Selection: Justifying the choice of GitHub as the primary data source and outlining the criteria for selecting repositories (size, update frequency).
Developer Activity Information: Defining which developer activity information to collect and how to extract it from repositories.
Software Similarity Measurement: Explaining the rationale for using vector representations (PVA) for software product descriptions and determining the appropriate similarity threshold.
Effort Estimate Calculation: Describing how to combine the effort values of similar software projects to arrive at a final estimate (using Walkerden's triangle function). "If the effort estimates of the first, second, and third closest neighbors are a, b, and c, respectively, then the effort required to develop z is expressed as Effort (z)= (3a + 2b + 1c )/6."

### 5. Experimental Evaluation

The paper details several experiments conducted to evaluate the performance of the proposed SDEE method, comparing it to existing techniques like ATLM (algorithmic), NeuralNet, ABE (Analogy-Based Estimation) and LOC (Lines of Code) straw man estimators. Key aspects of the experimental setup include:

Randomized Trials and k-Fold Cross-Validation: Employing these methods to assess the generalization ability of the models.
Evaluation Metrics: Using a range of metrics to evaluate performance, including:
Magnitude of Relative Error (MRE)
Mean Magnitude of Relative Error (MMRE)
Median Magnitude of Relative Error (MdMRE)
Mean Absolute Residual (MAR)
Logarithmic Standard Deviation (LSD)
RE*
Standardized Accuracy (SA): "SA represents how much better an effort estimation technique is than random guessing. The larger the value of SA, the better is the model performance."
Effect Size (Cohen's d, Hedges' g, Glass's Δ, Cliff's δ)
Statistical Significance Testing: Using t-tests, Cliff's delta tests, and parametric effect tests to validate the statistical significance of the results. Pearson's correlation coefficient is used to validate the correlation between the SDEE metrics.
Experiment #6: A test to validate the performance of the tool by professional programmers.
Results:
"Our system achieves the highest SA of 87.26% (Cliff’s δ = 0.88) when compared with the LOC straw man estimator and 42.7% (Cliff’s δ = 0.74) with the ATLM method at 99% confidence level (p < 0.00001)."
The tool is available at https://doi.org/10.21227/d6qp-2n13 and https://doi.org/10.5281/zenodo.5095723.

### 6. Threats to Validity

The authors acknowledge potential threats to the validity of their findings:

Internal Validity: Handling corner cases like refactoring and removing outliers from the dataset.
External Validity: The limitations of generalizing the results to software projects outside of the OSS context.
Statistical Conclusion Validity: Performing several statistical experiments with a 95% or greater confidence level to validate efficacy.

### 7. Conclusion and Future Work

The paper concludes that developer activity metrics, combined with PVA-based software similarity detection, can lead to more accurate and reliable SDEE. The authors suggest future work to include incremental effort estimates, developer characteristics, geographical location, and social interactions in the model.