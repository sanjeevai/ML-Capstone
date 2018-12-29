# Machine Learning Engineer Nanodegree

## Machine Learning Capstone

## Project: Factors affecting graduation and retention rates in the U.S. Colleges

## Table of Contents

1. [Project Overview](#overview)
2. [Problem Statement](#statement)
3. [Metrics](#metrics)
4. [Files](#files)
5. [References](#refs)

<a id='overview'></a>

### Project Overview

This is the third project in Term 2 of [Machine Learning Engineer Nanodegree](https://in.udacity.com/course/machine-learning-engineer-nanodegree--nd009t) from [Udacity](https://in.udacity.com/). In this project I have used supervised learning techniques to find the most relevant university level factors which affect retention and graduation rates in the U.S. colleges.

<a id='statement'></a>

### Problem Statement

In this problem, we will use supervised learning techniques to determine which university level factors are relevant in affecting the graduation and retention rates in the U.S. colleges. Variable names for graduation and retention rates are explained below:

- For graduation rates:

    **C150_4_POOLED_SUPP**

    Completion rate for first-time, full-time students at **four-year institutions** (150% of expected time to completion) , pooled in two-year rolling averages and suppressed for small n (<30) size.

- For retention rates:

    **RET_FT4**

    First-time, full-time student retention rate at **four-year institutions**.

These are just the target variables. The feature space consist of **100+ variables.** To know more about them, please see `metadata.xlsx` file.

<a id='metrics'></a>

### Metrics

We have used **r2 score** as the metric for performance of our model. In statistics, the coefficient of determination, denoted R2 or r2 and pronounced "R squared", is the proportion of the variance in the dependent variable that is predictable from the independent variable(s)<sup>[[1]](#ref1)</sup>. It provides a measure of how well observed outcomes are replicated by the model, based on the proportion of total variation of outcomes explained by the model<sup>[[2]](#ref2)[[3]](#ref3)[[4]](#ref4)</sup>.

r2 = 1 - RSS/TSS

here:
RSS = sum of squares of difference between actual values(yi) and predicted values(yi^),

TSS = sum of squares of difference between actual values (yi) and mean value (Before applying Regression).

So you can imagine TSS representing the best(actual) model, and RSS being in between our best model and the worst absolute mean model in which case we'll get RSS/TSS < 1. If our model is even worse than the worst mean model then in that case RSS > TSS(Since difference between actual observation and mean value < difference predicted value and actual observation)<sup>[[5]](#ref5)</sup>.

R squared is a good metric for this problem because this is a regression problem and this metric can provide a clear understanding of a regression model's performance by comparing the predicted value with true value in the simplest way.

In our problem we have 2 target variables, both continuous and scaled using `StandardScaler` function from sklearn. So, `r2_score` is a fit metric for this problem.

<a id='files'></a>

### Files

<pre>
.
|
+-data
| |
| +-+data.csv-------------# INPUT DATA WITH 123 VARIABLES AND 7593 OBSERVATIONS.
| +-+metadata.xlsx--------# EXPLANATION OF VARIABLES USED IN DATA. IT SHOW WHAT
| |                       # EACH VARIABLE STANDS FOR, WHAT IS THE DATA TYPE OF
| |                       # EACH VARIABLE, ETC.
| |
+-img---------------------# SAVED PLOTS FROM project.ipynb
|
+-proposal
| |
| +-+proposal.md----------# PROPOSAL WAS REQUIRED TO SUBMIT BEFORE THE PROJECT
| |                       # TO GIVE A PRACTICAL EXPERIENCE OF HOW TECHNICAL
| |                       # PROJECTS ARE CARRIED OUT.
| +-+proposal.pdf---------# PDF EXPORT OF proposal.md.
|
+-report
| |
| +-+report.md------------# IT SUMMARISES THE ENTIRE WORKFLOW OF THIS PROJECT.
| +-+report.pdf-----------# PDF EXPORT OF report.md
|
+-+project.ipynb----------# NOTEBOOK FOR DATA ANALYSIS AND MODEL IMPLEMENTATION.

</pre>

<a id='refs'></a>

### References

<a id="ref1"></a>

1. http://stattrek.com/statistics/dictionary.aspx?definition=coefficient_of_determination

<a id="ref2"></a>

2. Steel, R. G. D.; Torrie, J. H. (1960). Principles and Procedures of Statistics with Special Reference to the Biological Sciences. McGraw Hill.

<a id="ref3"></a>

3. Glantz, Stanton A.; Slinker, B. K. (1990). Primer of Applied Regression and Analysis of Variance. McGraw-Hill. ISBN 0-07-023407-8.

<a id="ref4"></a>

4. Draper, N. R.; Smith, H. (1998). Applied Regression Analysis. Wiley-Interscience. ISBN 0-471-17082-8.

<a id="ref5"></a>

5. https://stackoverflow.com/questions/23309073/how-is-the-r2-value-in-scikit-learn-calculated
