# FHFRM
 Functional Hybrid Factor Regression Model

This FHFRM package is developed by Chao Huang and Hongtu Zhu from the BIG-S2 lab. 

With the rapid growth of modern technology, many large-scale biomedical studies, e.g.,
Alzheimer’s disease neuroimaging initiative (ADNI) study, have been conducted to collect
massive datasets with large volumes of complex information from increasingly large cohorts.
Despite the numerous successes of biomedical studies, the imaging heterogeneity has posed
challenges in integrative analysis of data collected from multi-centers or multi-studies. The
study-level heterogeneity can result from the difference in study environment, population,
design, and protocols, which are mostly unknown. Surrogate variable analysis (SVA), which
is a powerful tool in tackling this heterogeneity, has been widely used in genomic studies.
However, the imaging data is usually represented as functional phenotype while no existing
SVA procedures work for functional responses. To address these challenges, a functional hybrid
factor regression model (FHFRM) is proposed to handle the unknown factors. Several
inference procedures are established for estimating the unknown parameters and detecting
the unknown factors. 

# Command Script 
We provide four scripts to run the data analysis with FHFRM. The usage of each script is listed below.

1. generate simulated curve data and covariates 

usage: python gen_data.py 1

where the augment '1' indicates the first scenario in our FHFRM paper, which means that hidden factors are independent with observed covariates. You can also enter other three different values for this augment: 2 (hidden factors are weakly correlated with observed covariates); 3 (hidden factors are moderatelycorrelated with observed covariates); and 4 (hidden factors are highly collrealted with observed covariates).

2. run main script

usage: python simu.py ./input/ ./result/

where './input/' is the directory to the simulates dataset; './result/' is the directory for saving all the result

3. summarize all the result

usage: python ./simu_summary.py ./result/

where './result/' is the directory for saving all the result
