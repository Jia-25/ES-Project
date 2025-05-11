"""
Team Members:
1. Wajeha Umer 2024659 - Team Lead 
2. Aafeen Gilaani 2024001
3. Aisha Noor 2024075

Dataset: Mental Health in Tech Survey (survey.csv)
Source: https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Load the dataset
df = pd.read_csv('survey.csv')

# Display basic information about the dataset
print("Dataset Information:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# Clean and prepare the data
df = df[(df['age'] >= 15) & (df['age'] <= 100)]
df = df.dropna(subset=['age'])  # Drop rows where age is Na

# =====================================================================
# CALCULATION OF MEAN AND VARIANCE, AND FREQUENCY DISTURBUTION
# =====================================================================

# 1.1 Calculate average and variance of the age data
age_mean = df['age'].mean()
age_var = df['age'].var(ddof=1)  # sample variance

print("\nBasic Statistics:")
print(f"Mean age (built in functions): {age_mean:.2f} years")
print(f"Variance of age (built in functions): {age_var:.2f} years")

# 1.2 Manual Calculation of average and variance of age
sum_age = 0
count = 0
for age in df['age']:
    sum_age += age
    count += 1
mean_age = sum_age / count
print(f"Mean age (Manual Calculation): {mean_age:.2f}")
sum_squared_diff = 0
for age in df['age']:
    sum_squared_diff += (age - mean_age) ** 2
variance_age = sum_squared_diff / (count - 1)
print(f"Variance (Manual Calculation): {variance_age:.2f}")

# Create age groups
age_bins = [15, 25, 35, 45, 55, 65, 100]
age_labels = ['15-24', '25-34', '35-44', '45-54', '55-64', '65+']
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)

# Get frequency distribution - sort by age group order
age_group_counts = df['age_group'].value_counts().sort_index()

# Frequency distribution
freq_dist = pd.DataFrame({
    'Age Group': age_group_counts.index,
    'Frequency': age_group_counts.values,
    'Relative Frequency': age_group_counts.values / age_group_counts.sum()
})

print("\nFrequency Distribution:")
print(freq_dist)

# Calculate statistics using properly aligned data
midpoints = [19.5, 29.5, 39.5, 49.5, 59.5, 82.5]  # Midpoints of each bin
freq_mean = np.sum(midpoints * age_group_counts) / age_group_counts.sum()
freq_var = np.sum(age_group_counts * (midpoints - freq_mean)**2) / (age_group_counts.sum() - 1)

print("\nStatistics from Frequency Distribution:")
print(f"Mean age: {freq_mean:.2f} years (original: {age_mean:.2f})")
print(f"Variance of age: {freq_var:.2f} (original: {age_var:.2f})")

# =====================================================================
# MANUAL STATISTICAL FUNCTIONS
# =====================================================================

# Manual t-distribution critical values (for 95% CI)
def get_t_critical_value(df, confidence=0.95):
    """Approximate t-critical values using lookup table"""
    # Simplified t-table for 95% confidence
    t_table = {
        1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57,
        6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23,
        15: 2.13, 20: 2.09, 30: 2.04, 40: 2.02, 
        60: 2.00, 120: 1.98, float('inf'): 1.96
    }
    
    # Find closest degrees of freedom
    closest_df = min(t_table.keys(), key=lambda x: abs(x - df))
    return t_table[closest_df]

# Manual chi-square critical values (for variance CI)
def get_chi2_critical_values(df, confidence=0.95):
    """Approximate chi-square critical values using lookup table"""
    # Simplified chi-square table for 95% CI
    chi2_table = {
        5: (0.831, 12.833),
        10: (3.247, 20.483),
        20: (9.591, 34.170),
        30: (16.047, 46.979),
        40: (24.433, 59.342),
        50: (32.357, 71.420),
        100: (77.929, 124.342)
    }
    
    closest_df = min(chi2_table.keys(), key=lambda x: abs(x - df))
    return chi2_table[closest_df]

# Manual t-test implementation
def manual_ttest(group1, group2, alternative='two-sided'):
    """Manual two-sample t-test assuming unequal variances"""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Calculate Welch's t-statistic
    t_stat = (mean1 - mean2) / np.sqrt(var1/n1 + var2/n2)
    
    # Welch-Satterthwaite degrees of freedom
    df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    
    # Calculate p-value (simplified approximation)
    if alternative == 'two-sided':
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))
    elif alternative == 'less':
        p_value = 0.5 * (1 + math.erf(t_stat / math.sqrt(2)))
    elif alternative == 'greater':
        p_value = 1 - 0.5 * (1 + math.erf(t_stat / math.sqrt(2)))
    
    return t_stat, p_value, df

# =====================================================================
# Confidence and tolerance intervals of Mean and Variance
# =====================================================================

# Splitting data into 80% training and 20% test
np.random.seed(42)
train_data = df.sample(frac=0.8, random_state=42)
test_data = df.drop(train_data.index)

n = len(train_data['age'])
train_mean = train_data['age'].mean()
train_std = train_data['age'].std(ddof=1)

# Manual 95% CI for mean
t_critical = get_t_critical_value(n-1)
margin = t_critical * (train_std / np.sqrt(n))
manual_conf_interval = (train_mean - margin, train_mean + margin)

# Manual 95% CI for variance
chi2_lower, chi2_upper = get_chi2_critical_values(n-1)
manual_var_conf_interval = ((n-1)*train_std**2/chi2_upper, (n-1)*train_std**2/chi2_lower)

# Manual 95% tolerance interval
z_critical = 1.96
k = z_critical * np.sqrt((n+1)/n)
manual_tol_interval = (train_mean - k*train_std, train_mean + k*train_std)

print("\nINTERVAL ESTIMATION RESULTS:")
print(f"95% CI for mean: ({manual_conf_interval[0]:.2f}, {manual_conf_interval[1]:.2f})")
print(f"95% CI for variance: ({manual_var_conf_interval[0]:.2f}, {manual_var_conf_interval[1]:.2f})")
print(f"95% Tolerance interval: ({manual_tol_interval[0]:.2f}, {manual_tol_interval[1]:.2f})")

# Validation with 20% test data
test_in_tolerance = ((test_data['age'] >= manual_tol_interval[0]) & 
                     (test_data['age'] <= manual_tol_interval[1]))
coverage_percentage = 100 * np.mean(test_in_tolerance)

print("\nValidation with 20% Test Data:")
print(f"Percentage of test data within tolerance interval: {coverage_percentage:.1f}%")

# =====================================================================
# Hypothesis testing
# =====================================================================
treatment_group = df.loc[df['treatment'] == 'Yes', 'age'].dropna()
no_treatment_group = df.loc[df['treatment'] == 'No', 'age'].dropna()

# Perform t-test
manual_t_stat, manual_p_value, manual_df = manual_ttest(
    treatment_group,
    no_treatment_group,
    alternative='two-sided'
)

print("\nMANUAL HYPOTHESIS TESTING:")
print("H0: Mean age of treatment seekers = Mean age of non-seekers")
print("H1: Mean age of treatment seekers ≠ Mean age of non-seekers")
print(f"n of Treatment group: {len(treatment_group)}")
print(f"n of Non-treatment group: {len(no_treatment_group)}")
print(f"Mean age - Treatment group: {treatment_group.mean():.2f}")
print(f"Mean age - Non-treatment group: {no_treatment_group.mean():.2f}")
print(f"Variance - Treatment group: {treatment_group.var():.2f}")
print(f"Variance - Non-treatment group: {no_treatment_group.var():.2f}")
print(f"Manual t-statistic: {manual_t_stat:.2f}, p-value: {manual_p_value:.4f}")

alpha = 0.05
if manual_p_value < alpha:
    print("Conclusion: Reject H0 - Significant age difference between groups")
else:
    print("Conclusion: Fail to reject H0 - No significant age difference between groups")

# =====================================================================
# VISUALIZATIONS
# =====================================================================

# 1. Age Distribution Histogram
plt.hist(df['age'], bins=6, edgecolor='black', color='skyblue')
plt.title('Age Distribution Histogram')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 2. Age Group Pie Chart
plt.pie(age_group_counts, labels=age_group_counts.index, autopct='%1.1f%%',
        colors=['gold', 'lightcoral', 'lightskyblue', 'lightgreen', 'violet', 'orange'])
plt.title('Age Group Distribution')
plt.show()

# 3. Gender Pie Chart
gender_counts = df['Gender'].value_counts().head(5)  # Top 5 genders
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%',
        colors=['lightblue', 'pink', 'lightgrey', 'lightgreen', 'violet'])
plt.title('Gender Distribution (Top 5)')
plt.show()

# 4. Treatment by Age Group
treatment_by_age = pd.crosstab(df['age_group'], df['treatment'], normalize='index') * 100
treatment_by_age.plot(kind='bar', stacked=True, ax=plt.gca(), 
                      color=['lightcoral', 'lightgreen'])
plt.title('Treatment Seeking by Age Group')
plt.ylabel('Percentage')
plt.xlabel('Age Group')
plt.legend(title='Sought Treatment', loc='upper right')
plt.show()
